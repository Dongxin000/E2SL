#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <chrono>
#pragma once
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


namespace kernels{
//pretty trivial kernel that has as many threads in the x-dimension of a block as output channels
//it is used as fallback!
    template<typename scalar_t>
    __global__ void cond_mul(
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
        //extern __shared__ uint8_t shared[];
        const int ind = blockIdx.x * blockDim.y + threadIdx.y;
        const int m = weights.size(1);//m input channels
        const int n = weights.size(2);//n output channels
        const int in = threadIdx.x;//index for the output is the thread index

        //Note that storing weights + accumulators in shared memory does not necessarily yield better results.
        //especially if the indices are very random
        if (ind >= input.size(0)) {
            return;
        }

        const int ind_w = inds[ind];
#ifdef FORCE_DOUBLE
        double result = bias[ind_w][0][in];
#else
        scalar_t result = bias[ind_w][0][in];
#endif
        for (int im = 0; im < m; im++) {
            result += input[ind][im] * weights[ind_w][im][in];
        }
        output[ind][in] = result;
    }


/*
A specialized kernel buffering 32 outputs. The weights are stored in registers and read in an alighned fashion.
 The same applies for the input.

how to use nvidia profiler: (profiling needs sudo rights but python doesn't find modules with sudo)
source venv/bin/activate
sudo env PATH=$PATH nvprof --analysis-metrics -f -o prof.nvvp venv/bin/python test_cuda_cond_mul.py
nvvp prof.nvvp

as the nvidia profiler deprecates newer gpus use the NSIGHT COMPUTE Profiler:
in the project directory run:
source venv/bin/activate
then:
sudo env PATH=$PATH /usr/local/cuda-11/bin/ncu --export profiled venv/bin/python test_cuda_cond_mul.py
maybe add stuff like:  --section MemoryWorkloadAnalysis for further investigation

The maximum occupancy is at 50%
Tho fix this:
at least 64 threads per block
reduce register count to 64, (for n=32 it is 94, n=16 it is 78 etc.)
shared memory can't co any higher than 4096 bytes

But as memory reads seem super efficient, this doesn't play a role as arithmetic and memory systems are
at 60 to 90%. A good value.

getting 2 warps to work simultaneously would be possible:
1) Each warp takes half of the pixel, keeping a separate set of weights in the registers.
    For n=32 and 94 registers, this yields 63% theoretical occupancy.
2) Split the output vector of size n in half. For n=32 this would result in approx 78 regs and 75% occupancy.
    We would also need to inefficiently read the weights. (each warp would always read 16 consecutive floats)
3) split the input vector (blocks) of size m in half. (n=32 -> ~78regs -> 75%)
    But the reads of the inputs would be less efficient. + we would need atomic adds for the accumulators.
*/
    template<typename scalar_t, int m_per_warp, int n_per_set> //TODO: this one is in use
    __global__ void
    cond_mul_cache_output(
            const int parts_in, //sets as in the template and parts as in this parameter are the same TODO: rename either set or part!
            const int parts_out,//run multiple times as accumulator can only store 32 pix with 32 channels max.
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
        extern __shared__ uint8_t shared[];

        const int base_ind = 32 * blockIdx.x; // the starting pixel for this block
        const int overall_samples = input.size(0);

        const int threads = 32; // threads in one block/warp (always 32)
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int simultaneous_pix = m_per_warp; //threads / n // same as blockDim.y
        const int warps_per_weight_set = n_per_set; // n

        //buffer for the weights of the n outputs (for big n this will use too many registers)
        scalar_t w[n_per_set];

        scalar_t *acc = (scalar_t *) &shared[0];
        //load indices
        int weight_index;
        if ((base_ind + tid) < overall_samples) { //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_index = inds[base_ind + tid];
        }
        //if n <=32 (n = n_per_set) we can load enough weights / have enough shared memory for the results of the next 32
        // outputs. If n is multiple of 32 we could just do it in multiple blocks/parts.
        for (int l = 0; l < parts_out; l++) {

            //clear the accumulator
            for (int i = 0; i < threads; i++) {
                acc[i * threads + tid] = 0;
            }
            //the input/weights of one pixel need to be split into parts
            for (int j = 0; j < parts_in; j++) {
                // load the next 32 values for 32 pixel:
                int last_ind = -1;
                for (int k = 0; k < 32; k++) {
                    int pix = base_ind + k;
                    if (pix >= overall_samples) {
                        break;
                    }
                    scalar_t v = input[pix][32 * j + tid];
                    int ind_w = __shfl_sync(0xffffffff, weight_index, k);

                    // in case of a new index for loading, we reload new weights
                    if (ind_w != last_ind) {

                        //we load a set of weights for n_per_set outputs
                        for (int i = 0; i < n_per_set; i++) {
                            int im = j * 32 + i * blockDim.y + threadIdx.y; //index along m direction of weight / input
                            w[i] = weights[ind_w][im][threadIdx.x + l * 32];
                        }
                        last_ind = ind_w;
                    }
                    scalar_t result = 0;

                    //v holds 32 values of the input features of this block.
                    for (int i = 0; i < n_per_set; i++) {
                        result += w[i] *
                                  __shfl_sync(0xffffffff, v, i * blockDim.y + threadIdx.y);
                    }
                    //page conflicts are not a big factor for performance.
                    //the complex indexing scheme with + n_per_set should prevent page conflicts
                    //int group_in_line = (k + threadIdx.y) % n_per_set;
                    int group_in_line = k;
                    acc[threadIdx.y * (32 * n_per_set) + // block within smem
                        n_per_set * group_in_line + // group within block
                        threadIdx.x] += result; //index within group

                }
            }
            // the warp should be in sync anyway (except for turing gpus and newer there it might differ!!!)
            __syncwarp();
            // n_per_set also means that one set has the size 32/n_per_set.
            // With 32 threads this means we need n_per_set iterations.
            for (int i = 0; i < n_per_set; i++) {
                int pix_local = i * blockDim.y + threadIdx.y;
                int pix = base_ind + pix_local;
                if (pix >= overall_samples) {
                    return;
                }
                int ind_w = __shfl_sync(0xffffffff, weight_index, pix_local);
                scalar_t accu = bias[ind_w][0][threadIdx.x];

                //iterate over all the accumulators for this set of values
                for (int j = 0; j < simultaneous_pix; j++) {
                    //non-wasteful way of resolving page conflicts
                    //int group_in_line = (threadIdx.y + i * blockDim.y + j) % n_per_set;
                    //non avoiding page conflicts is just as fast
                    int group_in_line = (threadIdx.y + i * blockDim.y);
                    accu += acc[j * (n_per_set * 32) + // block within smem
                                n_per_set * group_in_line + // group within block
                                threadIdx.x]; // index within group

                }
                output[pix][threadIdx.x + l * 32] = accu;

            }
        }

    }




/*
* all what has been written in the comments of the function above is applied here...
* it improves performance for n = 1, 2 and 4! For bigger n, it uses too many registers.
* to reach 100% occupancy it would be ideal to have 64 threads per block not more than 2048 bytes per block.
* the limit for shared memory is reached for n = 8 due to the additional index we use to prevent bank conflicts
* (otherwise it could be at n=16).
*/
    template <typename scalar_t,int m_per_warp,int n> //TODO: THIS IS USED
    __global__ void cond_mul_cache_output_high_occupancy( //cond_mul_cuda_forward_deep_reuse32_high_occupancy_kernel // cond_mul_cache_output_high_occupancy
            const int parts, //the parts in which the input/weights get loaded. for each part all outputs are calculated simultanously
            const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
            const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
            torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
        extern __shared__ uint8_t shared[];

        //each block has 64 threads in this version, each warp of 32 is processing 32 "pixel"
        const int base_ind = 64 * blockIdx.x + 32*threadIdx.z; // the starting pixel for this block
        const int overall_samples = input.size(0);
        const int m = weights.size(1);
        const int threads = 32; // threads in one block/warp (always 32)
        const int tid = threadIdx.y * blockDim.x + threadIdx.x; //thread index within a warp
        const int simultaneous_pix = m_per_warp; //threads / n // same as blockDim.y
        //const int columns_per_warp = m_per_warp; //threads / n
        const int warps_per_weight_set = n; // n

        //lets store weights for the processing in a register vairable
        scalar_t w[n];
        //the accumulator should be exactly n*32 long
#ifdef FORCE_DOUBLE
        double *acc = (double*)&shared[sizeof(double) * 32 * n * threadIdx.z];
#else
        scalar_t *acc = (scalar_t*)&shared[sizeof(scalar_t) * 32 * n * threadIdx.z]; //TODO: find out if threadIdx.z really is used!!!
#endif
        //load indices
        int weight_index;
        if( (base_ind + tid) < overall_samples){ //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_index = inds[base_ind + tid];
        }
        //scalar_t v[32];
        //clear the accumulator
        for(int i = 0;i < n; i++){
            acc[i * 32 + tid] = 0;
        }


        //the input/weights of each pixel need to be split into parts (of size 32)
        //TODO: think why we are buffering the weights and not input values + output accumulators?
        for(int j = 0; j < parts;j++){
            // load the next 32 values for 32 pixel:
            int last_ind = -1;
            for(int k = 0; k < 32; k++){
                int pix = base_ind + k;

                //stop when we are out of pixel!
                if( pix >= overall_samples){
                    break;
                }
                //load the value we need
                scalar_t v = input[pix][32 * j + tid];
                //get the current index we are at from other threads within the warp.(loaded in the beginning)
                int ind_w = __shfl_sync(0xffffffff, weight_index, k);
                if(ind_w != last_ind){
                    for(int i=0;i<n;i++){
                        //im consists of a few things
                        //the threadIdx.y... the "lane" of weights/values since only blockDim.y input channels can be processed at once
                        //i... as blockDim.y "lanes" are processed at once a block of 32 needs to be split into n (blockDim.x) parts
                        //j... the part we are at.
                        int im = j * 32 + i * blockDim.y + threadIdx.y; //index along m direction of weight / input
                        w[i] = weights[ind_w][im][threadIdx.x];
                    }
                    last_ind = ind_w;
                }
                scalar_t result = 0;
                for(int i=0;i<n;i++){
                    result += w[i] * __shfl_sync(0xffffffff, v, i * blockDim.y + threadIdx.y);
                }

                //proper reduction, we don't need to close down threads since they all are synced
                // (the unnecessary additions are not too bad)
                for (int offset = 16; offset >= n; offset /= 2)
                    result += __shfl_down_sync(0xffffffff, result, offset);

                if(tid < n){
                    //store result in accumulator (shared memory
                    acc[tid + k * n] += result;
                }

            }
        }
        __syncwarp(); // the warp should be in sync anyway (except for turing gpus... there it might differ!!!)

        for(int i=0;i<n;i++){
            int pix_local = i * blockDim.y + threadIdx.y;
            int pix = base_ind + pix_local;
            if(pix >= overall_samples){
                return;
            }
            int ind_w = __shfl_sync(0xffffffff, weight_index, pix_local);
            output[pix][threadIdx.x] = bias[ind_w][0][threadIdx.x] + acc[pix_local * n + threadIdx.x];
        }

    }

//TODO: fix the function names beginning with here!


//The opposite of the version from before, this is meant to double buffer the input
// or n=32 by splitting up the group via blockDim.z and adaption of shared memory and removal(templated if) of
// the syncthreads. More outputs n via more threads in anycase
// ACTUALLY: A SPECIALIZED VERSION WITHOUT SHARED MEMORY AND SHUFFELING INSTEAD WHEN N <= 32
// interesting case m=16 -> n=1 .... does not pay. It probably only makes sense for n being multiples of m!!!
    template <typename scalar_t, int m>//TODO: this is in use!
    __global__ void cond_mul_cache_input_double_buffer_few_in(//cond_mul_cuda_few_in_many_out( //cond_mul_cache_input_double_buffer_few_in
            const int loop_outer,
            const int loop_inner,
            const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
            const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
            torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
        const int tid = threadIdx.x + threadIdx.y * blockDim.x; //also this is ind_n
        const int ind_n = tid;
        const int base_ind = blockIdx.x * loop_outer * loop_inner;
        const int overall_samples = input.size(0);
        //shared memory needs to hold pixel_per_warp * input_channels pixel.
        extern __shared__ uint8_t shared[];
        scalar_t* buffers[2];
        buffers[0] = (scalar_t*)(&shared[0]);
        buffers[1] = (scalar_t*)(&shared[sizeof(scalar_t) * blockDim.x * blockDim.y]);
        int current_buffer = 1;

        int ind_w = -1;
        scalar_t w[m];
        scalar_t b;
        for(int i=0;i<loop_outer;i++){
            current_buffer = (current_buffer + 1) % 2;
            int pix_ind = base_ind + i * loop_inner + threadIdx.y;
            if( pix_ind < overall_samples){
                buffers[current_buffer][tid] = input[pix_ind][threadIdx.x];
            }
            __syncthreads();


            for(int j=0;j<loop_inner; j++){
                pix_ind = base_ind + i * loop_inner + j;

                if(pix_ind >= overall_samples){
                    return;
                }
                int ind_w_new = inds[pix_ind];
                if(ind_w_new != ind_w){
                    ind_w = ind_w_new;
                    b = bias[ind_w][0][ind_n];
                    scalar_t acc = b;
                    for(int k=0;k<m;k++){
                        //this actually is somewhat efficient when loading
                        w[k] = weights[ind_w][k][ind_n];
                        acc += w[k] * buffers[current_buffer][m * j + k];
                    }
                    output[pix_ind][ind_n]= acc;
                }else{
                    scalar_t acc = b;
                    for(int k=0;k<m;k++){
                        //reading from shared memory might not be the most efficient here
                        // there are 32 banks. we store 64 values in them. within one warp only 2 banks are used.
                        // each by 16 threads accessing the same value!
                        //printf("buffer_index: %d \n", m * j + k);
                        acc += w[k] * buffers[current_buffer][m * j + k];
                    }
                    //also this access is very efficient
                    output[pix_ind][ind_n]= acc;
                }
                //We can omit the synchronization due to double buffering
                //__syncthreads();

            }
        }
    }



//(unused, as it doesn't have speed benefits)
// in case we have fewer than 32 outputs we go by without any shared memory
//only worth if m=4,8,16 && n=1,2,4,8,16,32 &&  n >= m
// const int pixel_per_warp = 32;
//
//          const dim3 threads3(m, 32/m, 3); //96 threads (3 warps)
//          const int pixel_per_block = pixel_per_warp * threads3.z;
//          dim3 blocks((overall_samples + pixel_per_block - 1) / (pixel_per_block));
    template <typename scalar_t, int m>
    __global__ void cond_mul_reg_cache_weights_few_in(//cond_mul_cuda_few_in_many_out_no_shared( //cond_mul_reg_cache_weights_few_in
            const int loop_outer,// loop_outer * loop_inner must be 32
            const int loop_inner,// see ind_buffer! TODO: delete the remainders of ind_buffer
            const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
            const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
            torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {

        const int tid = threadIdx.x + threadIdx.y * blockDim.x; //index within warp!
        const int n = output.size(1);
        const int simultaneous_pixel_out = 32 / n; //how many pixel can one warp put out
        const int ind_n_out = tid % n; //basically reshaping block
        const int ind_pix_sub = tid/n;
        const unsigned mask = 0xffffffff;

        const int base_ind = (blockIdx.x * blockDim.z + threadIdx.z) * loop_outer * loop_inner;
        if(tid == 0){
            //printf("base_ind %d, outer_loop %d, inner_loop %d, simultaneous_pixel_out %d\n",base_ind,loop_outer, loop_inner, simultaneous_pixel_out);
        }
        const int overall_samples = input.size(0);
        scalar_t buffer;
        int ind_w = -1;
        int pix_ind = base_ind + tid;
        scalar_t w[m];
        scalar_t b;
        for(int i=0;i<loop_outer;i++){

            if(base_ind + i * loop_inner >= overall_samples){
                //if even the base index for this loop iteration is out of bounds
                // we want to exit since none of the threads in this warp will have work to do
                return;
            }
            pix_ind = base_ind + i * loop_inner + threadIdx.y;
            if( pix_ind < overall_samples){
                buffer = input[pix_ind][threadIdx.x]; //threadIdx = ind_m
            }
            //printf("pix_ind_read %d, ind_m %d, tid %d, warp_nr %d, blockIdx %d\n",pix_ind, threadIdx.x, tid, threadIdx.z, blockIdx.x);


            for(int j=0;j<loop_inner; j+=simultaneous_pixel_out){
                pix_ind = base_ind + i * loop_inner + j + ind_pix_sub;
                //printf("pixel_ind: %d of %d, j %d, loop_inner %d, simultaneous_pixel_out %d, local_pixel index %d \n", pix_ind, overall_samples,j,loop_inner,simultaneous_pixel_out,i * loop_inner + j + ind_pix_sub);

                int ind_w_new = ind_w;
                if(pix_ind < overall_samples){
                    ind_w_new = inds[pix_ind];
                }

                //it is somewhat unfortunate that the loop for loading the weights needs to
                //be separate from the one calculating the result
                if(ind_w_new != ind_w){// &&
                    //pix_ind < overall_samples){ //don't load new weights for non-existing pixel
                    ind_w = ind_w_new;
                    b = bias[ind_w][0][ind_n_out];
                    for(int k=0;k<m;k++) {
                        w[k] = weights[ind_w][k][ind_n_out];
                    }
                }
                scalar_t acc = b;
                for(int k=0;k<m;k++){
                    //it is important that none of the threads has exited for loading the things with shuffle
                    //printf("pixel_ind: %d of %d, locally: %d \n", pix_ind, overall_samples, k + (j + ind_pix_sub) * m);
                    scalar_t value = __shfl_sync(mask, buffer, k + (j + ind_pix_sub) * m);
                    acc += w[k] * value;
                }
                if(pix_ind < overall_samples) {
                    output[pix_ind][ind_n_out] = acc;
                }


            }
        }
    }

}//namespace kernels