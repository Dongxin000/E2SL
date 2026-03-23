#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <chrono>
#include "cond_mul_cuda_forward_kernel.cuh"
#include "cond_mul_cuda_experimental.cuh"
//#define FORCE_DOUBLE


namespace kernels{

template <bool set_size_at_0> //true for every but the very last
__global__ void integrate_start_inds(
        const size_t class_count,
        int32_t *counters,
        int stride){
    const int base_ind = 32 * (threadIdx.y + blockIdx.x * blockDim.y);
    const int lane = threadIdx.x;//warp lane
    const int ind = (lane + base_ind) * stride;
    if (ind>=class_count){
        return;
    }
    unsigned int mask = __activemask();
    int last_lane = __popc(mask) - 1;
    int count_self = counters[ind];
    int count;

    //we assume the neighbour to the left
    int sum = __shfl_up_sync(mask, count_self, 1);

    //the first lane should be zero (relative offset within this block)
    if(lane == 0){
        sum = 0; //this only happens once
    }

    for(int i=1;i<=16;i*=2){ //1,2,4,8,16
        int count_other = __shfl_up_sync(mask, sum, i);
        if (lane>=i){ //if the access to the left is out of bounds we do not add the value
            //printf("step %d, lane %d \n", i, lane);
            sum += count_other;
        }
    }

    if(set_size_at_0){
        //on the last lane the input value + the sum (of all values left of it) is what we want to put out on the
        //first lane
        int overall_sum = count_self + sum;
        //use highest active lane here!
        overall_sum =__shfl_sync(mask, overall_sum, last_lane); //the last lane contains the sum over all images
        if(lane == 0){
            sum = overall_sum;
        }
    }

    counters[ind] = sum;

}

__global__ void integrate_start_inds_back(
        const size_t class_count,
        int32_t *counters,
        int stride){
    const int base_ind = 32 * (threadIdx.y + blockIdx.x * blockDim.y);
    const int lane = threadIdx.x;//warp lane
    const int ind = (lane + base_ind) * stride;

    if(ind >= class_count){
        return;
    }
    int sum = counters[ind];
    const unsigned int full_mask = 0xffffffff;
    //get the offset from the very first element
    int offset = __shfl_sync(full_mask, sum, 0);
    if(lane != 0) {
        sum += offset;
    }
    counters[ind] = sum;



}

__global__ void count_classes(
                const size_t class_count,
                const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds,
                int32_t *counters){
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if(ind >= inds.size(0)){
        return;
    }
    int ind_w = inds[ind];
    if(ind_w >= class_count || ind_w < 0){
    	//TODO: put back in!
        printf("[count_classes]something is seriously off here ind_w %d, class_count %d \n",ind_w, class_count);
    }
    atomicAdd(&counters[ind_w], 1);
}

__global__ void setup_indices(
                const size_t class_count,
                const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds,
                const int32_t *sizes, // the amount of elements on each class
                const int32_t *start_inds, //the staring indices for each class in the lookup buffer
                int32_t *lookup_buffer,
                int32_t *counters
                ){
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if(ind >= inds.size(0)){
        return;
    }
    int ind_w = inds[ind];
    if(ind_w > class_count){
    	//TODO: put back in!!!
        printf("[setup_indices]something is seriously off here ind_w %d, class_count %d \n",ind_w, class_count);
    }
    //TODO: utilize warp aggregated atomics here!!!
    int count_old = atomicAdd(&counters[ind_w], 1);
    int start_ind = start_inds[ind_w];
    lookup_buffer[start_ind + count_old] = ind;

}

template <typename scalar_t>
__global__ void cond_mul_cuda_backward_b(
                        const int32_t *sample_count,
                        const int32_t *starting_inds,
                        const int32_t *lookup_buffer,
                        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
                        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_b){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int im = tid % grad_b.size(2);
    const int ind_w = tid / grad_b.size(2);
    if(ind_w >= grad_b.size(0)){
        return;
    }
#ifdef FORCE_DOUBLE
	double accu = 0;//TODO: get rid of the need for double precision operations here!
#else
	scalar_t accu = 0;
#endif
    const int start_ind = starting_inds[ind_w];
    const int count = sample_count[ind_w];
    for(int i=0; i < count; i++){
        int ind = lookup_buffer[start_ind + i];
        accu += grad_output[ind][im];
    }
    grad_b[ind_w][0][im] = accu;
}


template <typename scalar_t>
__global__ void cond_mul_cuda_backward_w(
                        const int32_t *sample_count,
                        const int32_t *starting_inds,
                        const int32_t *lookup_buffer,
                        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
                        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
                        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_w){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = grad_w.size(1);
    const int n = grad_w.size(2);
    const int imn = tid % (m * n); // this is not fast or beautiful
    const int ind_w = tid / (m * n);
    const int im = imn / n;
    const int in = imn % n;//this is not beautiful
    if(ind_w >= grad_w.size(0)){
        return;
    }
#ifdef FORCE_DOUBLE
    double accu = 0;//TODO: get rid of the need for double precision operations here!
#else
	scalar_t accu = 0;
#endif
    const int start_ind = starting_inds[ind_w];
    const int count = sample_count[ind_w];
    for(int i=0; i < count; i++){
        int ind = lookup_buffer[start_ind + i];
        // while grad_output will be read relatively efficiently (neighbouring threads read neighbouring values)
        // input is read less efficient. Alltogether it probably is not a superterrible approach, even when
        // the lookup_buffer introduces some pointer chasing.
        // What easily could be improved: Aligning the indices so that workgroups would be working on the same block
        // and have the same count of values to accumulate.
        accu += grad_output[ind][in] * input[ind][im];
    }
    grad_w[ind_w][im][in] = accu;
}
} // namespace kernels

std::vector<torch::Tensor> cond_mul_cuda_forward(
    torch::Tensor input,
    torch::Tensor inds,
    torch::Tensor weights,
    torch::Tensor bias) {
  auto options = torch::TensorOptions()
    .dtype(weights.dtype())
    .layout(torch::kStrided)
    .device(weights.device());
    //.requires_grad(true);
  auto output = torch::zeros({input.size(0), weights.size(2)}, options);

  const int overall_samples = input.size(0);
  const int m = weights.size(1);
  const int n = weights.size(2);
  //std::cout << "i shit you not!!! this is forward" << std::endl;




  AT_DISPATCH_FLOATING_TYPES(input.type(), "cond_mul_forward_cuda", ([&] {
      //const int threads = 32;
      //int simultaneous_pix = threads/n;
      //memory used for weights, bias, variables and accumulator
      //size_t shared_size = 0;

      if((((n == 1) || (n == 2) || (n == 4)) && m%32 == 0)){
          //this is one very common path as n=1 and m=32 is used for the regressor

          //neither is it good for n == 32 nor for n == 16 and for n == 1 its for sure not any better!
#ifdef FORCE_DOUBLE
		  size_t shared_size = 2 * sizeof(double) * 32 * n; // for the accumulator
#else
		  size_t shared_size = 2 * sizeof(scalar_t) * 32 * n; // for the accumulator
#endif

            const int per_group = 32/n;
            const dim3 threads3(n, per_group, 2); //lets have 64 threads per group (doubles the use of shared memory)
            const dim3 blocks((overall_samples + 64 - 1) / 64);
            const int parts = (m + 32 - 1) / 32;

            switch(n){
                case 1:
                    kernels::cond_mul_cache_output_high_occupancy<scalar_t, 32, 1><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 2:
                    kernels::cond_mul_cache_output_high_occupancy<scalar_t, 16, 2><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 4:
                    kernels::cond_mul_cache_output_high_occupancy<scalar_t, 8, 4><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                default:
                    std::cout << "argh! this is a trap" << std::endl;
            }
      }else if((((n == 8) || (n == 16) || (n == 32)) && m%32 == 0)) {
          //std::cout << "the most important branch" << std::endl;
          //this is a common branch as n=8, and n=16 with m=32 is used for the classifiers
          //shared memory (approx 4kb used) is used for the accumulator of the weights.

          //only using 32 threads per block means we have low occupancy (50%)
          // the memory bandwith still is well utilized with 70 to 85% though.
          const int threads = 32;
          // shared memory for the accumulator
          size_t shared_size = sizeof(scalar_t) * threads * (threads);

          const int per_group = 32 / n;
          const dim3 threads3(n, per_group);
          const dim3 blocks((overall_samples + 32 - 1) / 32);
          const int parts_in = (m + 32 - 1) / 32;
          const int parts_out = (n - 1) / 32 + 1; //in this branch, parts_out should always be 1 (it is meant for n>32)
          switch (n) {
              case 8:
                  kernels::cond_mul_cache_output<scalar_t, 4, 8> << < blocks, threads3, shared_size >> > (
                          parts_in,
                                  parts_out,
                                  input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                  inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
                  break;
              case 16:
                  kernels::cond_mul_cache_output<scalar_t, 2, 16> << < blocks, threads3, shared_size >> > (
                                  parts_in,
                                  parts_out,
                                  input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                  inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
                  break;
              case 32:
                  kernels::cond_mul_cache_output<scalar_t, 1, 32> << < blocks, threads3, shared_size >> > (
                                  parts_in,
                                  parts_out,
                                  input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                  inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
                  break;
              default:
                  //n is multiple of 32, the solution to that is to simply loop this kernel a bit!
                  std::cout << "argh! this is a trap" << std::endl;
          }
      }else if(((m==1 || m==2 || m==4 || m==8 || m==16)  &&
                (n==64 || n==96 || n==128 || n==160 || n==192 || n==224 || n==256))){ // with 32 still as fast as next
          //unused branch but somewhat optimal

          const int pixel_per_block = std::max(64, n/m);
          dim3 blocks((overall_samples + pixel_per_block - 1) / (pixel_per_block)); // most of the blocks take the next 32 pixel for each active warp (64)
          const dim3 threads3(m, n/m); //alltogether we want n threads.
          size_t shared_size = sizeof(scalar_t) * threads3.x * threads3.y * threads3.z * 2; //two times since we double buffer!!!


          switch(m){
              case 1:
                  kernels::cond_mul_cache_input_double_buffer_few_in<scalar_t, 1><<<blocks, threads3, shared_size>>>(
                            pixel_per_block /threads3.y, //how often do we need to read channels/pixels to fill all pixel
                                  threads3.y,//loop_inner (or how many pixel are read simultaneously by all the threads)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 2:
                  kernels::cond_mul_cache_input_double_buffer_few_in<scalar_t, 2><<<blocks, threads3, shared_size>>>(
                          pixel_per_block /threads3.y, //how often do we need to read channels/pixels to fill all pixel
                                  threads3.y,//loop_inner (or how many pixel are read simultaneously by all the threads)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 4:
                  kernels::cond_mul_cache_input_double_buffer_few_in<scalar_t, 4><<<blocks, threads3, shared_size>>>(
                          pixel_per_block /threads3.y, //loop outer
                                  threads3.y,//loop_inner (or how many pixel are read simultaneously by all the threads)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 8:
                  kernels::cond_mul_cache_input_double_buffer_few_in<scalar_t, 8><<<blocks, threads3, shared_size>>>(
                                    pixel_per_block / threads3.y, //loop outer
                                  threads3.y,//loop_inner (or how many pixel are read simultaneously by all the threads)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 16:
                  kernels::cond_mul_cache_input_double_buffer_few_in<scalar_t, 16><<<blocks, threads3, shared_size>>>(
                                  pixel_per_block /threads3.y, //loop outer
                                  threads3.y,//loop_inner (or how many pixel are read simultaneously by all the threads)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              default:
                  std::cout << "argh! this is a trap" << std::endl;
          };

      }else{
          //a more generic fallback branch!
          //std::cout << "fallback branch" << std::endl;
         //uncached version all values are fetched when needed!
         size_t per_group = 256/n;// it actually doesn't matter if this were 32 threads. works just the same
         assert(n * per_group == 256);
         const dim3 threads3(n, per_group);
         const dim3 blocks((overall_samples + per_group - 1) / per_group);


          kernels::cond_mul<scalar_t><<<blocks, threads3>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
            weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
      }
  }));

  //debug measures:
  //gpuErrchk( cudaPeekAtLastError() );
  //gpuErrchk( cudaDeviceSynchronize() );
  return {output};
}


//#define MEASURE_TIME
std::vector<torch::Tensor> cond_mul_cuda_backward(
        torch::Tensor grad_output,//gradient of output
        torch::Tensor input,
        torch::Tensor inds,
        torch::Tensor weights) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto device = weights.device();
    auto grad_weights = torch::zeros_like(weights);
    auto grad_input = torch::zeros_like(input);
    auto weights_t = torch::transpose(weights, 1, 2);
    weights_t = weights_t.contiguous(); //depending on the value of m and n this mostly brings speedups!
    auto options =
    torch::TensorOptions()
        .dtype(weights.dtype())
        .layout(torch::kStrided)
        .device(weights.device());
    //.requires_grad(true);

    auto grad_bias = torch::zeros({weights.size(0), 1, weights.size(2)}, options);

    auto bias_back_zero = torch::zeros({weights.size(0), 1, weights.size(1)}, options);
    size_t overall_samples = input.size(0);

    if(grad_output.size(0) == 0){
        //this is not supposed to happen but it happened once so lets keep it here
        assert(0);
    }

    int32_t *sizes_gpu;
    int32_t *starting_inds_gpu;
    int32_t *counters_gpu;
    int32_t *ind_lookup_gpu;

    cudaMalloc(&sizes_gpu, sizeof(int32_t) * weights.size(0));
    cudaMalloc(&starting_inds_gpu, sizeof(int32_t) * weights.size(0));
    cudaMalloc(&counters_gpu, sizeof(int32_t) * weights.size(0));
    cudaMalloc(&ind_lookup_gpu, sizeof(int32_t) * grad_output.size(0));

    cudaMemset(sizes_gpu, 0, sizeof(int32_t) * weights.size(0));
    cudaMemset(counters_gpu, 0, sizeof(int32_t) * weights.size(0));

    //count occurence of each class
    int threads = 256;
    dim3 blocks((overall_samples + threads - 1) / threads);

	//std::cout << "class_count ( count classes) " << weights.size(0) << std::endl;
    kernels::count_classes<<<blocks, threads>>>(weights.size(0),//nr of different classes //grad_output.size(0),
                                    inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),
                                    sizes_gpu); //the counts for each class
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		const char* errstr = cudaGetErrorString(error);
		std::cout << errstr << std::endl;
	}
    /******************************************************************************************/
    //To generate starting indices, we accumulate the sample counts for each class.
    // Algorithm works as follows:
    // 1. load in the counts in groups of 32 and shuffle up the count -> store such that lane 0 stores the overall count
    // 2. repeat at a increasing (x32) strides and set the very first index to 0
    //   The first entry in each group will have the overall count.
    //   The consecutive strides will have the relative offset from there.
    // 3. with decreasing (/32) strides update the relative offsets to absolute ones. Repeat till stride is zero
    int class_count = weights.size(0);
    //int32_t *starting_inds_gpu_2;
    //cudaMalloc(&starting_inds_gpu_2, sizeof(int32_t) * class_count);
    cudaMemcpy(starting_inds_gpu, sizes_gpu, sizeof(int32_t) * class_count, cudaMemcpyDeviceToDevice);
    int step_size = 1;
    //std::cout << "class_count " << class_count << std::endl;
    while ( (class_count / step_size) > 32 ){
        int groups_per_block = 4;
        dim3 threads3(32, groups_per_block);
        dim3 blocks3((class_count/step_size + threads3.x*threads3.y*threads3.z - 1) /  (threads3.x*threads3.y*threads3.z));
        //std::cout << "up stride:" << step_size << "blocks " << blocks3.x << std::endl;
        kernels::integrate_start_inds<true><<<blocks3, threads3>>>(
                class_count,
                starting_inds_gpu,
                step_size); //stride
        step_size *= 32;
    }
    dim3 threads3(32, 1);
    dim3 blocks3((class_count/step_size + threads3.x*threads3.y*threads3.z - 1) /  (threads3.x*threads3.y*threads3.z));
    // run kernel one last time! (don't write sum at first element!)
	dim3 blocks_int((overall_samples));

    kernels::integrate_start_inds<false><<<blocks3, threads3>>>(
            class_count,
            starting_inds_gpu,
            step_size); //stride


	while( step_size >= 32){
	    step_size /= 32;
	    //first action already is one step size below.
	    //TODO: add offsets until step size equals 1
        int groups_per_block = 4;
        dim3 threads3(32, groups_per_block);
        dim3 blocks3((class_count/step_size + threads3.x*threads3.y*threads3.z - 1) /  (threads3.x*threads3.y*threads3.z));

        kernels::integrate_start_inds_back<<<blocks3, threads3>>>(
                                                               class_count,
                                                               starting_inds_gpu,
                                                                step_size); //stride
	}


    //setup lookup buffer
    kernels::setup_indices<<<blocks, threads>>>( weights.size(0),//nr of different classes
                                    inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),
                                    sizes_gpu,
                                    starting_inds_gpu,
                                    ind_lookup_gpu,
                                    counters_gpu); // the counters for each individiual class

#ifdef MEASURE_TIME
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time for preparation: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    begin = end;
#endif

    std::vector<torch::Tensor>  result =  cond_mul_cuda_forward(
            grad_output,
            inds,
            weights_t,
            bias_back_zero);
    grad_input = result[0];
#ifdef MEASURE_TIME
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    std::cout << "Time for backward: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    begin = end;
#endif


    //calc gradients for input, w and b
    AT_DISPATCH_FLOATING_TYPES(weights.type(), "cond_mul_forward_cuda", ([&] {
        //gradient for input: basically the opposite of the forward path with transposed weights and zeroed bias
        const int m = weights_t.size(1);
        const int n = weights_t.size(2);


        // gradient for b
        threads = 256;
        dim3 blocks((grad_bias.size(0) * grad_bias.size(2) + threads - 1) / threads);
        kernels::cond_mul_cuda_backward_b<scalar_t><<<blocks, threads>>>(
                    sizes_gpu,
                    starting_inds_gpu,
                    ind_lookup_gpu,
                    grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    grad_bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());


#ifdef MEASURE_TIME
        cudaDeviceSynchronize();
        end = std::chrono::steady_clock::now();
        std::cout << "Time for accumulating bias grads: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        begin = end;
#endif

        threads = 256;
        blocks.x = (grad_weights.size(0) * grad_weights.size(1) * grad_weights.size(2) + threads - 1) / threads;
        kernels::cond_mul_cuda_backward_w<scalar_t><<<blocks, threads>>>(
                    sizes_gpu,
                    starting_inds_gpu,
                    ind_lookup_gpu,
                    grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    grad_weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());

    }));

#ifdef MEASURE_TIME
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    std::cout << "Time for accumulating weight grads: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
#endif

    //free all the buffers we created
    cudaFree(sizes_gpu);
    cudaFree(starting_inds_gpu);
    cudaFree(ind_lookup_gpu);
    cudaFree(counters_gpu);

  return {grad_input, grad_weights, grad_bias};
}
