/*
 ============================================================================
 Name        : juliaset.cu
 Author      : Wolfgang
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */

/*
__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize)
		data[idx] = 1.0/data[idx];
}
*/

/**
 * Host function that copies the data and launches the work on GPU
 */
/*
float *gpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	float *gpuData;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	
	static const int BLOCK_SIZE = 32;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	return rc;
}
*/

__device__ void cadd(float a_real, float a_imag, float b_real, float b_imag, float* c_real, float* c_imag)
{
    //Complex c;
    *c_real = a_real + b_real;
    *c_imag = a_imag + b_imag;
    //return c;
}

__device__ void cmul(float a_real, float a_imag, float b_real, float b_imag, float* c_real, float* c_imag)
{
    //Complex c;
    *c_real = (a_real*b_real)-(a_imag*b_imag);
    *c_imag = (a_real*b_imag)+(a_imag*b_real);
}

__device__ void cbetr(float a_real, float a_imag, float* btr)
{
    *btr = sqrt(pow(a_real,2)+pow(a_imag,2));
}

__device__ void qadd(float a_real, float a_i, float a_j, float a_k, float b_real, float b_i, float b_j, float b_k, float* c_real, float* c_i, float* c_j, float* c_k)
{
	*c_real = a_real + b_real;
	*c_i = a_i + b_i;
	*c_j = a_j + b_j;
	*c_k = a_k + b_k;
}

__device__ void qmul(float a_real, float a_i, float a_j, float a_k, float b_real, float b_i, float b_j, float b_k, float* c_real, float* c_i, float* c_j, float* c_k)
{
	*c_real = (a_real*b_real)-(a_i*b_i)-(a_j*b_j)-(a_k*b_k);
	*c_i = (a_real*b_i)-(a_i*b_real)-(a_j*b_k)-(a_k*b_j);
	*c_j = (a_real*b_j)-(a_i*b_k)-(a_j*b_real)-(a_k*b_i);
	*c_k = (a_real*b_k)-(a_i*b_j)-(a_j*b_i)-(a_k*b_real);
}

__device__ void qbetr(float a_real, float a_i, float a_j, float a_k, float* btr)
{
	*btr = sqrt(pow(a_real, 2)+pow(a_i, 2)+pow(a_j, 2)+pow(a_k, 2));
}

__global__ void calc_CJuliaset(float* A, float c_real, float c_imag, int number_of_iterations, float x_start, float y_start, float granularity, int N, int M)
{
	// Block index
	//int bx = blockIdx.x;
	//int by = blockIdx.y;

	// Thread index
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	//int y = blockDim.y * blockIdx.y + threadIdx.y;
	int n = x/N;
	int m = x-(n*N);


	int step=0;
	float temp_real=0;
	float temp_imag=0;
	float z_real=x_start-(n*granularity);
	float z_imag=y_start-(m*granularity);

	while(step < number_of_iterations)
	{
		cmul(z_real, z_imag, z_real, z_imag, &temp_real, &temp_imag);    //function to calculate JuliaSet  z(1) = z(0)² + c
	    cadd(temp_real, temp_imag, c_real, c_imag, &z_real, &z_imag);
	    cbetr(z_real, z_imag, &temp_real);
	    if(temp_real>2)
	    {
	        break;       //is NOT considered as in JuliaSet
	    }
	    //z.real=temp.real;
	    //z.imag=temp.imag;
	    step++;
	 }

	A[x] = step;
}


__global__ void calc_CMandelbrot(int* A, int number_of_iterations, float x_start, float y_start, float granularity, int N, int M)
{
	// Block index
	//int bx = blockIdx.x;
	//int by = blockIdx.y;

	// Thread index
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	//int y = blockDim.y * blockIdx.y + threadIdx.y;
	int n = x/N;
	int m = x-(n*N);


	int step=0;
	float temp_real=0;
	float temp_imag=0;
	float c_real=x_start-(n*granularity);//0;
	float c_imag=y_start-(m*granularity);//0;
	float z_real=0;//x_start-(n*granularity);
	float z_imag=0;//y_start-(m*granularity);

	while(step < number_of_iterations)
	{
		cmul(z_real, z_imag, z_real, z_imag, &temp_real, &temp_imag);    //function to calculate JuliaSet  z(1) = z(0)² + c
	    cadd(temp_real, temp_imag, c_real, c_imag, &z_real, &z_imag);
	    cbetr(z_real, z_imag, &temp_real);
	    if(temp_real>2)
	    {
	        break;       //is NOT considered as in JuliaSet
	    }
	    //z.real=temp.real;
	    //z.imag=temp.imag;
	    step++;
	 }

	A[x] = step;
}




__global__ void calc_CMandelbrot(int* A, long number_of_iterations, int N, int M)
{
	// Block index
	//int bx = blockIdx.x;
	//int by = blockIdx.y;

	// Thread index
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	//int y = blockDim.y * blockIdx.y + threadIdx.y;
	int n = x/N;
	int m = x-(n*N);
	float gran_n = 4./(M-1);
	float gran_m = 4./(N-1);

	int step=0;
	float temp_real=0;
	float temp_imag=0;
	float c_imag=2-(n*gran_n);//0;
	float c_real=-2+(m*gran_m);//0;
	float z_real=0;//x_start-(n*granularity);
	float z_imag=0;//y_start-(m*granularity);

	while(step < number_of_iterations)
	{
		cmul(z_real, z_imag, z_real, z_imag, &temp_real, &temp_imag);    //function to calculate MandelbrotSet  z(1) = z(0)² + c
	    cadd(temp_real, temp_imag, c_real, c_imag, &z_real, &z_imag);
	    cbetr(z_real, z_imag, &temp_real);
	    if(temp_real>2)
	    {
	        break;       //is NOT considered as in MandelbrotSet
	    }
	    //z.real=temp.real;
	    //z.imag=temp.imag;
	    step++;
	 }

	A[x] = step;
}


__global__ void calc_CMandelbrot(int* A, long number_of_iterations, int pixel_x, int pixel_y, float start_x, float start_y, float end_x, float end_y, int device)
{
	// Block index
	//int bx = blockIdx.x;
	//int by = blockIdx.y;

	// Thread index
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	//int y = blockDim.y * blockIdx.y + threadIdx.y;
	int n = x/pixel_x;
	int m = x-(n*pixel_x);
	float gran_n = 4./(pixel_y-1);
	float gran_m = 4./(pixel_x-1);

	int step=0;
	float temp_real=0;
	float temp_imag=0;
	float c_imag=2-(n*gran_n);//0;
	float c_real=-2+(m*gran_m);//0;
	float z_real=0;//x_start-(n*granularity);
	float z_imag=0;//y_start-(m*granularity);

	while(step < number_of_iterations)
	{
		cmul(z_real, z_imag, z_real, z_imag, &temp_real, &temp_imag);    //function to calculate MandelbrotSet  z(1) = z(0)² + c
	    cadd(temp_real, temp_imag, c_real, c_imag, &z_real, &z_imag);
	    cbetr(z_real, z_imag, &temp_real);
	    if(temp_real>2)
	    {
	        break;       //is NOT considered as in MandelbrotSet
	    }
	    //z.real=temp.real;
	    //z.imag=temp.imag;
	    step++;
	 }

	A[x] = step;
}



void startCalc(float start_x, float start_y, float end_x, float end_y, int pixel_x, int pixel_y, int iterations)
{
	clock_t prgstart, prgende;
	int O = 1;
	size_t size = pixel_x * pixel_y * O * sizeof(int);
	//int *d_A;
	int devices = 0;

	cudaError_t error;
	error = cudaGetDeviceCount(&devices);
	if (error != cudaSuccess)
	{
	    printf("cudaGetDeviceCount returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}
	printf("DeviceCount: %d\n", devices);

	int **h_A = (int**)malloc(devices); //(int *)malloc(size);
	int **d_A = (int**)malloc(devices);

	int i;
	for(i=0; i<devices; i++)
	{
		d_A[i] = (int*)malloc(size/devices);
		error = cudaSetDevice(i);

		if (error != cudaSuccess)
		{
		    printf("cudaSetDeviceCount returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		    exit(EXIT_FAILURE);
		}

		error = cudaMalloc((void **) &d_A[i], size/devices);

		if (error != cudaSuccess)
		{
		    printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		    exit(EXIT_FAILURE);
		}

		int BLOCK_SIZE = 512;
		int blockCount = (pixel_x*pixel_y)/BLOCK_SIZE;
		prgstart=clock();

		//calc_CJuliaset<<<blockCount, BLOCK_SIZE>>>(d_A, 0.1, 0.1, 1000, 0.5, 0.5, 0.1, N, M);
		//calc_CMandelbrot<<<blockCount, BLOCK_SIZE>>>(d_A, 1000, 0.5, 0.5, 0.01, N, M);
		calc_CMandelbrot<<<blockCount, BLOCK_SIZE>>>(d_A[i], 10000, pixel_x, pixel_y);

		cudaMemcpy(h_A[i], d_A[i], size, cudaMemcpyDeviceToHost);
		cudaFree(d_A);

		prgende=clock();//CPU-Zeit am Ende des Programmes
		printf("Laufzeit %.2f Sekunden\n",(float)(prgende-prgstart) / CLOCKS_PER_SEC);


		int a=0;
		for(a=0; a<pixel_x*pixel_y; a++)
		{

			if(a%pixel_x==0)
			{
				printf("\n");
			}

			if(h_A[i][a]<1000)
			{
				//printf("x");
			}
			else
			{
				//printf("o");
			}

			printf(" %d ", h_A[i][a]);
		}
	}

}



int main(int argc, char* argv[])
{
	/*
	clock_t prgstart, prgende;

	int N = 64;
	int M = 64;
	int O = 1;
	size_t size = N * M * O * sizeof(int);

	int *h_A = (int *)malloc(size);

	int *d_A;
	int devices = 0;

	cudaError_t error;
	error = cudaGetDeviceCount(&devices);

	if (error != cudaSuccess)
		{
		    printf("cudaGetDeviceCount returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		    exit(EXIT_FAILURE);
		}

	printf("DeviceCount: %d\n", devices);
	error = cudaMalloc((void **) &d_A, size);
	*/
	//error = cudaMalloc3D((void **) &d_A, size);
/*
	if (error != cudaSuccess)
	{
	    printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	int BLOCK_SIZE = 512;
	int blockCount = (M*N)/BLOCK_SIZE;//(size+BLOCK_SIZE-1)/BLOCK_SIZE;
	prgstart=clock();

	error = cudaSetDevice(0);
	if (error != cudaSuccess)
	{
	    printf("cudaSetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}
	*/
	//calc_CJuliaset<<<blockCount, BLOCK_SIZE>>>(d_A, 0.1, 0.1, 1000, 0.5, 0.5, 0.1, N, M);
	//calc_CMandelbrot<<<blockCount, BLOCK_SIZE>>>(d_A, 1000, 0.5, 0.5, 0.01, N, M);
	/*
	calc_CMandelbrot<<<blockCount, BLOCK_SIZE>>>(d_A, 10000, N, M);

	cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);

	prgende=clock();//CPU-Zeit am Ende des Programmes
	printf("Laufzeit %.2f Sekunden\n",(float)(prgende-prgstart) / CLOCKS_PER_SEC);

	int a=0;
	for(a=0; a<N*M; a++)
	{

		if(a%N==0)
		{
			printf("\n");
		}

		if(h_A[a]<1000)
		{
			//printf("x");
		}
		else
		{
			//printf("o");
		}

		printf(" %d ", h_A[a]);
	}
*/


	/*
	float *recCpu = cpuReciprocal(data, WORK_SIZE);
	float *recGpu = gpuReciprocal(data, WORK_SIZE);
	float cpuSum = std::accumulate (recCpu, recCpu+WORK_SIZE, 0.0);
	float gpuSum = std::accumulate (recGpu, recGpu+WORK_SIZE, 0.0);

*/
	/* Verify the results */
	//std::cout<<"gpuSum = "<<gpuSum<< " cpuSum = " <<cpuSum<<std::endl;

	/* Free memory */
	//delete[] data;
	//delete[] recCpu;
	//delete[] recGpu;

	startCalc(-2,2,-2,2,64,64,1024);
	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

