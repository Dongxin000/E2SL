//
// Created by simon on 30.03.22.
//

namespace kernels{



// Unused kernel that only gives a small speedup (30%) with small inputs m=1,2,4 and bigger outputs
//call when:
//    (((m == 1) && ((n==4) || (n == 8) || (n==16) || (n%32 == 0) )) || //n==1,2 would also be an inefficient option
//    ((m == 2) && ((n==4)|| (n==8) || (n%16 == 0))) || //n==1,2 would probably be very inefficient
//    ((m == 4) && ((n==2) || (n==4) || (n%8 == 0))))
//with
//          size_t per_group = 32 / m;
//          const dim3 threads3(m, per_group, 2); //64 threads 2 active warps per block
//          const int pixel_per_warp = 32;
//
//          int simultaneous_output_pixel = std::max(1, static_cast<int>(threads3.y / n));
//          int simultaneous_output_channels = threads3.y /
//                                             simultaneous_output_pixel;
//          int loop_pixel = pixel_per_warp / simultaneous_output_pixel;
//          dim3 blocks((overall_samples + 2 * pixel_per_warp - 1) /
//                      (2 * pixel_per_warp)); // most of the blocks take the next 32 pixel for each active warp (64)
//          size_t shared_size =
//                  sizeof(scalar_t) * m * pixel_per_warp * 2;
//again we want to have maximum occupancy here:
//maximum 2048 bytes of shared memory per block
//maximum 32registers per thread
// 64 threads per block
// To always keep the shared memory at its full potential one could store more than 32 pixel simultaneously
// calculation for shared memory:
// 1 channel input: 512 vals buffer -> 2048 bytes (weight index array would take 16 registers)
// 2 channel input: 256 vals buffer -> 2048 bytes (weight index array would take 8 registers)
// 4 channel input: 128 vals buffer -> 2048 bytes (weight index array would take 4 registers)
// 8 channel input: 64 vals buffer -> 2048 bytes (weight index array would take 2 registers)
// 16 channel input: 32 vals buffer -> 2048 bytes (weight index takes 1 register)
// 32 channel input: 16 vals buffer -> 2048 bytes (weight index array takes 1 register for half of the warps
// for the applications this was designed, we don't at best have 608 pixel per line with 16 different classes
// this gives 38 consecutive pixel with same class. Probably the improvements of implementing an index array are minor.
//TODO: maybe also make this work with more than a 32 pixel buffer. (it doesn't pay
//TODO: for the very common case of m=16 and n=1 it would be nice to actually calculate multiple
// pixel outputs at the same time! (rename pixel_per_warp to something more fitting since we are looping trough pixel)
// also, the neighbouring pixel should be looped as in working on pixel in the order (0,16) (1,17) (2,18)

//TODO: m_per_warp and n_per_warp is the same as blockDim.x and blockDim.y
    template <typename scalar_t, int pixel_per_warp> //TODO:THIS IS IN USE
    __global__ void cond_mul_cache_input_few_in(//cond_mul_cuda_forward_deep_reuse32_few_in_many_out(//cond_mul_cache_input_few_in
            const int loop_pixel, //in the normal case we would loop over pixel_per_warp pixel since we go 1 pixel at a time TODO: rename loop_out_pixel
            const int simultaneous_out_pixel, //if we have very few output channels we can handle multiple pixel at once
            const int simultaneous_out_channels, //TODO: simultaneous out_channels can be derived from blockDim.y (blockDim.y = simultaneous_out_pixel * simultaneous_out_channels)
            const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
            const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
            torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
        //return;
        //shared memory needs to hold pixel_per_warp * input_channels pixel.
        extern __shared__ uint8_t shared[];
        const int base_ind = pixel_per_warp * (blockIdx.x * blockDim.z + threadIdx.z); // the starting pixel for this block
        const int overall_samples = input.size(0);
        const int m_per_warp = blockDim.x; // same as m

        const int n = output.size(1);
        //const int m = input.size(1);//todo:would m_per_warp be the same minus the cost of reading it?

        //split up the available output channels (threadIdx.y) into the simultaneous pixel into the simultaneously
        // worked channels (ind_y_sub) and the simultaneously worked pixel (ind_y_main)
        // e.g. m=2 -> blockDim = (2,16,2) and 32 pixel_per_warp -> simultaneous_out_channels = 16
        // ind_y goes 0,1,2,3,4,5,6,7, 08,09,10,11,12,13,14,15
        // ind_y_main 0,0,0,0,0,0,0,0, 01,01,01,01,01,01,01,01 // main is used for the input channels m
        // ind_y_sub  0,1,2,3,4,5,6,7, 00,01,02,03,04,05,06,07 // sub is used for the input dimension n
        int ind_y_main = threadIdx.y / simultaneous_out_channels;
        int ind_y_sub = threadIdx.y % simultaneous_out_channels;
        //TODO: is this the same as


        //buffer the input
        scalar_t* buf_in = (scalar_t*)&shared[sizeof(scalar_t) * threadIdx.z * pixel_per_warp * m_per_warp];
        //printf("offset shared: %d \n", (threadIdx.z * pixel_per_warp * m_per_warp));
        const int tid = threadIdx.y * blockDim.x + threadIdx.x; //thread index within a warp

        //TODO: if we want to buffer more than 32 pixel (could make sense) we want weight index to be an array.
        //get the current index
        int weight_index;
        if( (base_ind + tid) < overall_samples){ //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_index = inds[base_ind + tid];
        }

        //load the the input of the next pixel_per_warp pixel into shared memory
        //parts_in has to be selected accordingly when calling this kernel
        for(int i=0;i<pixel_per_warp;i+=blockDim.y){
            //blockDim.y tells me how many pixel we load simultaneously
            //blockDim.x how many weights are loaded for each pixel (should be the size of m)
            const int ind = base_ind + threadIdx.y + i;

            if(ind >= overall_samples){
                break;
            }
            //we don't need to trick around here since it doesn't seem like we are provoking bank conflicts
            //tid goes from 0:32t warps, i goes for e.g. m=16 ->blockDim = (16,2,2) and 32 pixel_per_warp
            // 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, ...30 multiplied by blockDim.y this makes 0, 32, 64, 96 etc. till
            // 480 (512-32) Note that 32 pixel_per_warp are using too much shared memory for m=16... in reality it would be
            // 16 for m=16 and 8 for m=32
            buf_in[tid + i * blockDim.x] = input[ind][threadIdx.x];
        }
        //return;
        //accumulate over all the elements and create outputs:
        //TODO: parts_out could be skipped if we go for something like this:
        // for(int i=0; i < n; i+=simultaneous_out_channels)
        //for(int i=0; i < parts_out; i++){
        for(int i=0; i < n; i+=simultaneous_out_channels){
            //the index for the output channel
            //int ind_n = ind_y_sub + i * simultaneous_out_channels;//threadIdx.y + i * n_per_warp;
            int ind_n = ind_y_sub + i;
            int ind_m = threadIdx.x;
            //iterate over the next pixel_per_warp (32 in most cases) pixel!
            int ind_w_last = -1;
            scalar_t b = 0;
            scalar_t w = 0;
            //TODO: if we want ot have a buffer for more than 32 pixel we need to take care of this!
            for(int j=0; j<loop_pixel;j++){ //iterate over the 16/32 pixel
                //int pix_ind = base_ind + j; //TODO: fix this for working multiple pixel simultaneously:
                int pix_ind = base_ind + j + loop_pixel * ind_y_main;
                //pix_ind = base_ind + j + simultaneous_out_pixel * ind_y_main; //TODO: find out if loop_pixel / simultaneous out_pixel does the same here?
                //if(pix_ind >= overall_samples){
                //TODO: now we have one issue: not all pixel of this group work on one pixel...
                // we can't just break here and ignore all the potential shfl_syncs that will never be achieved!!!!!!
                //    break; // obviously we don't go further than the amount of samples we need
                //}
                int ind_w = __shfl_sync(0xffffffff, weight_index, j + ind_y_main * loop_pixel);
                if( ind_w_last != ind_w &&
                    (pix_ind < overall_samples)){ //prevent readout at invalid pixel!
                    ind_w_last = ind_w;
                    if(ind_n < n){ //in cases we have fewer outputs than we need threads for (e.g. 1 input channel and 16 output channels)
                        //printf("weight index, m, n%d, %d, %d\n", ind_w, ind_m, ind_n);
                        b = bias[ind_w][0][ind_n]; // even though only one thread (the one writing out really needs this)
                        w = weights[ind_w][ind_m][ind_n]; // this access is not the most efficient
                    }
                }
                //TODO: fix readout or right buffer!!! (index is definitely not right) (2022: is it still?)
                scalar_t acc = w * buf_in[(j + loop_pixel * ind_y_main) * m_per_warp + threadIdx.x];
                //accumulate via shuffeling (e.g. when we have 16 input channels and 2 outputs: the offsets should be
                // 8, 4, 2 and 1. the results of thread 0 and 16 should then be stored in the two outputs!!!)
                //during this phase only half of the threads are really active (so we have to make it up with occupancy)
                for (int offset = m_per_warp / 2; offset >= 1; offset /= 2)
                    acc += __shfl_down_sync(0xffffffff, acc, offset);
                if((threadIdx.x == 0) && (ind_n < n) && // only the one of the threads is supposed to writeout results
                   (pix_ind < overall_samples)) {// prevent invalid writeouts
                    output[pix_ind][ind_n] = acc + b;
                    //printf("%d, %d, = %f\n", pix_ind, ind_n, (float)(acc + b));
                }
            }


        }

    }

// (UNUSED)
// Buffering input and output in smem. Demonstration how inefficient this is! (3 times as slow)
// It is meant to run on inputs of m and n being multiple of 32
// Due to inefficient memory access it is 3 times slower than the other optimized paths working on
// per-warp versions.
// const dim3 threads3(256); //256 threads per block
// size_t smem_size = 16384;
// size_t pixel_per_block = smem_size / (sizeof(scalar_t) * (m + n) + sizeof(scalar_t));
    template<typename scalar_t>
    __global__ void cond_mul_buffer_io_kernel(
            const int cache_count, // how many inputs/outputs and indices are cached in shared memory
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> indices,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {

        const int overall_pix_count = input.size(0);
        const int tid = threadIdx.x; //index within warp!
        const int n = output.size(1);
        const int m = input.size(1);
        const int base_ind = blockIdx.x * cache_count;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            //printf("m %d, n %d \n", m, n);
        }
        extern __shared__ uint8_t shared[];
        scalar_t *in = (scalar_t *) shared;
        scalar_t *out = &in[cache_count * m];
        int *inds = (int *) (&out[cache_count * n]);

        //load input:
        int i = tid;
        while (i < m * cache_count) {
            int ch = i % m;
            int ind = i / m; // pixel index(local
            if (ind + base_ind >= overall_pix_count) {
                break;
            }
            in[ind * m + ch] = input[ind + base_ind][ch];
            i += blockDim.x;
        }
        //load inds:
        i = tid;
        while (i < cache_count) {
            if (base_ind + i >= overall_pix_count) {
                break;
            }
            inds[i] = indices[base_ind + i];
            i += blockDim.x;
        }
        //setup output buffer
        i = tid;
        while (i < n * cache_count) {
            int ch = i % n;
            int ind = i / n; // pixel index(local
            if (ind + base_ind >= overall_pix_count) {
                break;
            }
            out[ind * n + ch] = 0;
            i += blockDim.x;
        }

        __syncthreads();
        //do processing
        //this part eats up a lot of compute (actually it is 3 times as slow as the m=32 to n=32 path)
        int ind_w = -1;
        float w;
        i = tid;
        while (i < m * n) {
            int ind = 0;
            //for a transposed weight tensor:
            // (just transposing the weight tensor via
            // torch::transpose(weights, 1, 2) and weightsT = weightsT.contiguous();
            // takes 1.5 times as long as a good implementation)
            //int ind_m = i % m;
            //int ind_n = i / m;
            int ind_m = i / n;
            int ind_n = i % n;

            while (ind < cache_count &&
                   (ind + base_ind) < overall_pix_count) {
                if (ind_w != inds[ind]) {
                    ind_w = inds[ind];
                    w = weights[ind_w][ind_m][ind_n];
                }
                scalar_t update = w * in[ind * m + ind_m];
                //out[ind * n + ind_n] += update; (not using the atomic is wrong and only 1/6th faster)
                atomicAdd(&out[ind * n + ind_n], update);
                ind++;
            }
            i += blockDim.x;
        }

        //write out!
        __syncthreads();
        i = tid;
        while (i < n * cache_count) {
            int ch = i % n;
            int ind = i / n; // pixel index(local
            if (ind + base_ind >= overall_pix_count) {
                break;
            }
            int ind_w = inds[ind];
            output[ind + base_ind][ch] = out[ind * n + ch] + bias[ind_w][0][ch];
            i += blockDim.x;
        }
    }


    //unused as it takes twice as long as the kernel used in the final software
    //it uses only 40 registers and has near 100% occupancy
    //special case for conditional multiply for m=32 and n=32 128 threads per block.
    //we load the input of 4 pixel into shared memory, load the m*n weights if needed,
    // do the multiplications and finally write everything into a 4 pixel output buffer.
    // optimizations:
    // 1) As it only uses 40 registers  per thread one could easily use
    //    8 registers per thread to buffer the weights in registers.
    //    (the weights are used by the same threads as they are fetched with)
    // 2) Double/multi- buffer the inputs and outputs.
    //    This could go as far as that every thread fetches the input from a ring buffer
    //    but the first one encountering a missing input needs to read it from VRAM.
    //    (obviously taking care that no other thread still needs it)
    //    for the output and weights the scheme has to be similarly awful.
    //    probably this will use too many registers
    // 3) Warp level specialization. 2 warps loading data, 2 threads doing the computations?
    //    synchronization and double buffering will probably eat a lot of registers making them inefficient
    //    also only half the threads would be utilizing their compute
    // call by:
    // size_t shared_size = (32*32 + 128 + 128) * sizeof(scalar_t);
    // const dim3 threads3(32, 4); //128 threads
    // const dim3 blocks((overall_samples + 32 - 1) / 32);
    template<typename scalar_t>
    __global__ void
    cond_mul_32_32_smem_io_weights(
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
        extern __shared__ uint8_t shared[];
        const int threads = 128; // threads in one block/warp (always 32)
        scalar_t *in_buf = (scalar_t*)shared;
        scalar_t *w = &in_buf[threads];
        scalar_t *out_buf = &w[32 * 32];
        //128 + 32*32 + 128


        const int base_ind = 32 * blockIdx.x; // the starting pixel for this block
        const int overall_samples = input.size(0);
        //blockDim.x = 32 and blockDim.y = 4
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        //load indices
        int weight_inds;
        if ((base_ind + threadIdx.x) < overall_samples) { //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_inds = inds[base_ind + threadIdx.x];
        }

        int weight_ind = -1;
        for(int i=0;i<8;i++){
            //load in 4 pixel!
            int ind_local = i * blockDim.y + threadIdx.y;
            int ind = base_ind + ind_local;
            if(ind < overall_samples){
                in_buf[tid] = input[ind][threadIdx.x];
                out_buf[tid] = 0;//TODO: maybe load the bias here already?
            }
            //wait for all threads to finish loading the new inputs
            __syncthreads();//this syncthread could be omitted by double buffering io

            //calculate the results for 4 pixel
            for(int j=0;j<4;j++){

                //in case we encounter a new index, we load new weights
                ind_local = i * blockDim.y + j;
                int weight_ind_new  =__shfl_sync(0xffffffff, weight_inds, ind_local);
                if(weight_ind_new != weight_ind){
                    //wait for all threads to finish the use of the old weights
                    //__syncthreads();
                    weight_ind = weight_ind_new;
                    int im = threadIdx.y; //index m axis
                    while(im<32){
                        int in = threadIdx.x; //index n axis
                        //int in_fixed = (in + k) % 32; // get rid of bank conflicts
                        w[im * blockDim.x + in] = weights[weight_ind][im][in]; // index, m, n
                        im += blockDim.y; //+= 4;
                    }
                    //wait for all threads to finish loading new weights (before multiplying with them)
                    //__syncthreads();
                }

                //multiply
                int in = threadIdx.x; // index output
                int im = threadIdx.y;
                scalar_t update = 0;
                while(im < 32){
                    //int in_fixed = (in + k) % 32; // get rid of bank conflicts
                    update += in_buf[j * blockDim.x + im] * w[im * blockDim.x + in];

                    im += blockDim.y;
                }
                atomicAdd(&out_buf[j * blockDim.x + in], update);

            }
            //wait for all threads to finishe the multiplicaton
            __syncthreads(); //this syncthread could be omitted by double buffering
            //write out the output for these 4 pixel
            ind_local = i * blockDim.y + threadIdx.y; // pixel index within this group of 32
            if(ind < overall_samples){
                int weight_ind_new  =__shfl_sync(0xffffffff, weight_inds, ind_local);
                int in = threadIdx.x;
                output[ind][in] = out_buf[tid] + bias[weight_ind_new][0][in];//TODO: maybe load the bias here already?
            }
        }
    }

    //same as above but using registers instead of the shared memory for the weights
    // is almost as fast as the fastest 32 to 32 implementation (twice as fast as version above)
    //optimization ideas:
    //1) is the input buffering necessary? get rid of first __syncthreads() yes it is
    //2) bigger output buffer (such that there is almost no need for fewer syncthreads for output
    //3) only 2 instead of 4 warps?
    template<typename scalar_t>
    __global__ void
    cond_mul_32_32_smem_io(
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
        extern __shared__ uint8_t shared[];
        const int threads = 128; // threads in one block/warp (always 32)

        // each thread only pushes one input value to this buffer (128)
        scalar_t *in_buf = (scalar_t*)shared;

        // for each output of 4 pixel we have 4 warps working on part of the problem.
        // this needs to hold 32 channels, for 4 pixel and 4 warps 32*4*4=512 before accumulating
        scalar_t *out_buf = &in_buf[threads];

        scalar_t w[8];


        const int base_ind = 32 * blockIdx.x; // the starting pixel for this block
        const int overall_samples = input.size(0);
        //blockDim.x = 32 and blockDim.y = 4
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        //load indices
        int weight_inds;
        if ((base_ind + threadIdx.x) < overall_samples) { //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_inds = inds[base_ind + threadIdx.x];
        }

        int weight_ind = -1;
        for(int i=0;i<8;i++){
            //load in 4 pixel!
            int ind_local = i * blockDim.y + threadIdx.y;
            int ind = base_ind + ind_local;
            if(ind < overall_samples){
                in_buf[tid] = input[ind][threadIdx.x];
            }
            //wait for all threads to finish loading the new inputs
            __syncthreads();//this syncthread could be weakened by double buffering io

            //calculate the results for 4 pixel
            for(int j=0;j<4;j++){

                //in case we encounter a new index, we load new weights
                ind_local = i * blockDim.y + j;
                ind = base_ind+ind_local;
                if(ind >= overall_samples){
                    break;
                }
                int weight_ind_new  =__shfl_sync(0xffffffff, weight_inds, ind_local);
                if(weight_ind_new != weight_ind){
                    weight_ind = weight_ind_new;
                    for(int k=0;k<8;k++){
                        int im = threadIdx.y + k * 4; //index m axis
                        int in = threadIdx.x;
                        w[k] = weights[weight_ind][im][in];
                    }

                }

                //multiply
                int in = threadIdx.x; // index output
                scalar_t update = 0;
                for(int k=0;k<8;k++){
                    int im = threadIdx.y + k * 4; //index m axis
                    update += in_buf[j * blockDim.x + im] * w[k];
                }
                //atomicAdd(&out_buf[j * blockDim.x + in], update);
                out_buf[j * (32 * 4) + threadIdx.y * 32 + in] = update;


            }
            //wait for all threads to finishe the multiplicaton
            __syncthreads(); //this syncthread could be weakened by double buffering
            //write out the output for these 4 pixel
            ind_local = i * blockDim.y + threadIdx.y; // pixel index within this group of 32
            if(ind < overall_samples){
                int weight_ind_new  =__shfl_sync(0xffffffff, weight_inds, ind_local);
                int in = threadIdx.x;
                //output[ind][in] = out_buf[tid] + bias[weight_ind_new][0][in];//TODO: maybe load the bias here already?
                //new
                scalar_t accu = bias[weight_ind_new][0][in];
                for(int k=0;k<4;k++){
                    accu += out_buf[threadIdx.y * 32 * 4 + k*32 + in];
                }
                output[ind][in] = accu;
            }
        }
    }

    //same as above without buffering the input (it is slightly slower than the other)
    template<typename scalar_t>
    __global__ void
    cond_mul_32_32_smem_output(
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
        extern __shared__ uint8_t shared[];
        const int threads = 128; // threads in one block/warp (always 32)

        // for each output of 4 pixel we have 4 warps working on part of the problem.
        // this needs to hold 32 channels, for 4 pixel and 4 warps 32*4*4=512 before accumulating
        scalar_t *out_buf = (scalar_t*)shared;

        scalar_t w[8];


        const int base_ind = 32 * blockIdx.x; // the starting pixel for this block
        const int overall_samples = input.size(0);
        //blockDim.x = 32 and blockDim.y = 4
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        //load indices
        int weight_inds;
        if ((base_ind + threadIdx.x) < overall_samples) { //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_inds = inds[base_ind + threadIdx.x];
        }

        int weight_ind = -1;
        for(int i=0;i<8;i++){
            //calculate the results for 4 pixel
            for(int j=0;j<4;j++){
                int ind = base_ind + i * 4 + j;
                if(ind >= overall_samples){
                    break;
                }
                scalar_t in_vals = input[ind][threadIdx.x];
                //in case we encounter a new index, we load new weights
                int ind_local = i * blockDim.y + j;
                int weight_ind_new  =__shfl_sync(0xffffffff, weight_inds, ind_local);
                if(weight_ind_new != weight_ind){
                    weight_ind = weight_ind_new;
                    for(int k=0;k<8;k++){
                        int im = threadIdx.y + k * 4; //index m axis
                        int in = threadIdx.x;
                        w[k] = weights[weight_ind][im][in];
                    }

                }

                //multiply
                int in = threadIdx.x; // index output
                scalar_t update = 0;
                for(int k=0;k<8;k++){
                    int im = threadIdx.y + k * 4; //index m axis
                    scalar_t val = __shfl_sync(0xffffffff, in_vals, im);
                    update += val * w[k];
                }
                //atomicAdd(&out_buf[j * blockDim.x + in], update);
                out_buf[j * (32 * 4) + threadIdx.y * 32 + in] = update;


            }
            //wait for all threads to finishe the multiplicaton
            __syncthreads(); //this syncthread could be weakened by double buffering
            //write out the output for these 4 pixel
            int ind_local = i * blockDim.y + threadIdx.y; // pixel index within this group of 32
            int ind = base_ind + ind_local;
            if(ind < overall_samples){
                int weight_ind_new  =__shfl_sync(0xffffffff, weight_inds, ind_local);
                int in = threadIdx.x;
                //output[ind][in] = out_buf[tid] + bias[weight_ind_new][0][in];//TODO: maybe load the bias here already?
                //new
                scalar_t accu = bias[weight_ind_new][0][in];
                for(int k=0;k<4;k++){
                    accu += out_buf[threadIdx.y * 32 * 4 + k*32 + in];
                }
                output[ind][in] = accu;
            }
        }
    }

    //same as above but just use 64 threads instead of 128
    template<typename scalar_t>
    __global__ void
    cond_mul_32_32_smem_output_3(
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
        extern __shared__ uint8_t shared[];
        const int threads = 64; // threads in one block/warp (always 32)

        // for each output of 4 pixel we have 4 warps working on part of the problem.
        // this needs to hold 32 channels, for 4 pixel and 4 warps 32*4*4=512 before accumulating
        scalar_t *out_buf = (scalar_t*)shared;

        scalar_t w[16];


        const int base_ind = 32 * blockIdx.x; // the starting pixel for this block
        const int overall_samples = input.size(0);
        //blockDim.x = 32 and blockDim.y = 4
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        //load indices
        int weight_inds;
        if ((base_ind + threadIdx.x) < overall_samples) { //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_inds = inds[base_ind + threadIdx.x];
        }

        int weight_ind = -1;
        for(int i=0;i<16;i++){
            //calculate the results for 2 pixel
            for(int j=0;j<2;j++){
                int ind = base_ind + i * 2 + j;
                if(ind >= overall_samples){
                    break;
                }
                scalar_t in_vals = input[ind][threadIdx.x];
                //in case we encounter a new index, we load new weights
                int ind_local = i * blockDim.y + j;
                int weight_ind_new  =__shfl_sync(0xffffffff, weight_inds, ind_local);
                if(weight_ind_new != weight_ind){
                    weight_ind = weight_ind_new;
                    for(int k=0;k<16;k++){
                        int im = threadIdx.y + k * 2; //index m axis
                        int in = threadIdx.x;
                        w[k] = weights[weight_ind][im][in];
                    }

                }

                //multiply
                int in = threadIdx.x; // index output
                scalar_t update = 0;
                for(int k=0;k<16;k++){
                    int im = threadIdx.y + k * 2; //index m axis
                    scalar_t val = __shfl_sync(0xffffffff, in_vals, im);
                    update += val * w[k];
                }
                //atomicAdd(&out_buf[j * blockDim.x + in], update);
                out_buf[j * (32 * 2) + threadIdx.y * 32 + in] = update;


            }
            //wait for all threads to finishe the multiplicaton
            __syncthreads(); //this syncthread could be weakened by double buffering
            //write out the output for these 2 pixel
            int ind_local = i * blockDim.y + threadIdx.y; // pixel index within this group of 32
            int ind = base_ind + ind_local;
            if(ind < overall_samples){
                int weight_ind_new  =__shfl_sync(0xffffffff, weight_inds, ind_local);
                int in = threadIdx.x;
                //output[ind][in] = out_buf[tid] + bias[weight_ind_new][0][in];//TODO: maybe load the bias here already?
                //new
                scalar_t accu = bias[weight_ind_new][0][in];
                for(int k=0;k<2;k++){
                    accu += out_buf[threadIdx.y * 32 * 2 + k*32 + in];
                }
                output[ind][in] = accu;
            }
        }
    }

    //same as the one above but with buffered inputs! (almost as fast as needed)
    //also there is almost no way to improve it as it already uses 62 registers
    template<typename scalar_t>
    __global__ void
    cond_mul_32_32_smem_io_2(
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
        extern __shared__ uint8_t shared[];
        const int threads = 64; // threads in one block/warp (always 32)

        // each thread only pushes one input value to this buffer (64)
        scalar_t *in_buf = (scalar_t*)shared;

        // for each output of 2 pixel we have 2 warps working on part of the problem.
        // this needs to hold 32 channels, for 4 pixel and 4 warps 32*2*2=128 before accumulating
        scalar_t *out_buf = &in_buf[threads];

        //this strategy is a register hog! it feels as if this takes more than just 16 registers.
        scalar_t w[16];

        const int base_ind = 32 * blockIdx.x; // the starting pixel for this block
        const int overall_samples = input.size(0);
        //blockDim.x = 32 and blockDim.y = 4
        const int tid = threadIdx.y * 32 + threadIdx.x;

        //load indices
        int weight_inds;
        if ((base_ind + threadIdx.x) < overall_samples) { //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_inds = inds[base_ind + threadIdx.x];
        }

        int weight_ind = -1;
        for(int i=0;i<16;i++){

            //load in 2 pixel!
            int ind_local = i * 2 + threadIdx.y;
            int ind = base_ind + ind_local;
            if(ind < overall_samples){
                in_buf[tid] = input[ind][threadIdx.x];
            }
            //wait for all threads to finish loading the new inputs
            //these syncthreads barely cost performance
            __syncthreads();

            //calculate the results for 2 pixel
            for(int j=0;j<2;j++){
                int ind = base_ind + i * 2 + j;
                //in case we encounter a new index, we load new weights
                int ind_local = i * 2 + j;
                int weight_ind_new  =__shfl_sync(0xffffffff, weight_inds, ind_local);
                if(weight_ind_new != weight_ind){
                    weight_ind = weight_ind_new;
                    for(int k=0;k<16;k++){
                        int im = threadIdx.y + k * 2; //index m axis
                        int in = threadIdx.x;
                        w[k] = weights[weight_ind][im][in];
                    }

                }

                //multiply
                int in = threadIdx.x; // index output
                scalar_t update = 0;
                for(int k=0;k<16;k++){
                    int im = threadIdx.y + k * 2; //index m axis
                    update += in_buf[j * 32 + im] * w[k];
                }
                //atomicAdd(&out_buf[j * blockDim.x + in], update);
                out_buf[j * (32 * 2) + threadIdx.y * 32 + in] = update;


            }
            
            //wait for all threads to finish the multiplicaton
            __syncthreads(); //this syncthread could be weakened by double buffering
            //write out the output for these 2 pixel
            ind_local = i * 2 + threadIdx.y; // pixel index within this group of 32
            ind = base_ind + ind_local;
            if(ind < overall_samples){
                int weight_ind_new  =__shfl_sync(0xffffffff, weight_inds, ind_local);
                int in = threadIdx.x;
                scalar_t accu = bias[weight_ind_new][0][in];
                for(int k=0;k<2;k++){
                    accu += out_buf[threadIdx.y * 32 * 2 + k*32 + in];
                }
                output[ind][in] = accu;
            }


        }
    }



}