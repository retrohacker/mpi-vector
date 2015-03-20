#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int TAG_LENGTH = 0;
int TAG_VEC_A = 1;
int TAG_VEC_B = 2;
int TAG_SCALE = 3;
int TAG_SCALE_RESULT = 4;

void get_input(int** vector_a, int** vector_b, int* length, int* scalar);
void distribute_input(int* vector_a, int* vector_b, int* length, int* scalar);
void recv_input(int** vector_a, int** vector_b, int* length, int* scalar);
int compute_dot(int* vector_a, int* vector_b, int length);
void compute_scale(int* vector_a, int** scale, int length, int scalar);
void send_scale(int *scale, int length);
void gather_scale(int** scale, int length);
void e(int error);
int my_rank;
int comm_sz;
MPI_Comm comm;

/* Program entry point */
int main(int argc, char *argv[]) {
  int* vector_a;
  int length;
  int* vector_b;
  int scalar;
  int dot;
  int dot_result;
  int* scale;

  e(MPI_Init(&argc, &argv));
  comm = MPI_COMM_WORLD;
  e(MPI_Comm_size(comm,&comm_sz));
  e(MPI_Comm_rank(comm,&my_rank));
  if(my_rank == 0) {
    get_input(&vector_a,&vector_b,&length,&scalar);
    distribute_input(vector_a,vector_b,&length,&scalar);
    dot = 0;
    gather_scale(&scale,length);
  } else {
    recv_input(&vector_a,&vector_b,&length,&scalar);
    dot = compute_dot(vector_a,vector_b,length);
    compute_scale(vector_a,&scale,length,scalar);
    send_scale(scale,length);
  }
  MPI_Reduce(&dot,&dot_result,1,MPI_INT,MPI_SUM,0,comm);
  if(my_rank == 0) {
    printf("Dot Product of A and B: %d\n",dot_result);
    if(length>0) {
      int i;
      printf("A * Scalar: <%d",scale[0]);
      for(i = 1; i < length; i++) {
        printf(",%d",scale[i]);
      }
      printf(">\n");
    }
  }
  free(vector_a);
  free(vector_b);
  free(scale);
  MPI_Finalize();
  return 0;
}

void get_input(int** vector_a, int** vector_b, int* length, int* scalar) {
  int i;

  printf("Vector Length: ");
  scanf("%d",length);
  *vector_a = malloc(sizeof(int) * (*length));
  *vector_b = malloc(sizeof(int) * (*length));
  for(i = 0; i < *length; i++) {
    printf("Value %d of Vector A: ",i);
    scanf("%d",&(*vector_a)[i]);
  }
  for(i = 0; i < *length; i++) {
    printf("Value %d of Vector B: ",i);
    scanf("%d",&(*vector_b)[i]);
  }
  printf("Enter a scalar: ",scalar);
  scanf("%d",scalar);
}

void distribute_input(int* vector_a, int* vector_b, int* length, int* scalar) {
  int i;
  int index = 0;
  int block_sz = *length / (comm_sz - 1);
  int r = *length % (comm_sz - 1);

  for(i = 1; i < comm_sz;i++) {
    int bs = block_sz;
    if(i <= r) bs+=1;
    e(MPI_Send(&bs,1,MPI_INT,i,TAG_LENGTH,comm));
    e(MPI_Send(&vector_a[index],bs,MPI_INT,i,TAG_VEC_A,comm));
    e(MPI_Send(&vector_b[index],bs,MPI_INT,i,TAG_VEC_B,comm));
    e(MPI_Send(scalar,1,MPI_INT,i,TAG_SCALE,comm));
    index+=bs;
  }
}

void recv_input(int** vector_a, int** vector_b, int* length, int* scalar) {
  MPI_Status status;
  e(MPI_Recv(length,1,MPI_INT,0,0,comm,&status));
  *vector_a = malloc(sizeof(int) * *length);
  *vector_b = malloc(sizeof(int) * *length);
  e(MPI_Recv(*vector_a,*length,MPI_INT,0,1,comm,&status));
  e(MPI_Recv(*vector_b,*length,MPI_INT,0,2,comm,&status));
  e(MPI_Recv(scalar,1,MPI_INT,0,3,comm,&status));
}

int compute_dot(int* vector_a,int* vector_b, int length) {
  int result = 0;
  int i;
  for(i = 0; i < length; i++) {
    result+= vector_a[i] * vector_b[i];
  }
  return result;
}

void compute_scale(int* vector_a, int** scale, int length, int scalar) {
  *scale = malloc(sizeof(int) * length);
  int i;
  for(i=0;i<length;i++) {
    (*scale)[i] = vector_a[i] * scalar;
  }
}

void send_scale(int* scale, int length) {
  e(MPI_Send(scale,length,MPI_INT,0,TAG_SCALE_RESULT,comm));
}

void gather_scale(int** scale, int length) {
  int i;
  int index = 0;
  int block_sz = length / (comm_sz - 1);
  int r = length % (comm_sz - 1);
  MPI_Status status;
  *scale = malloc(sizeof(int) * length);
  for(i = 1; i < comm_sz; i++) {
    int bs = block_sz;
    if(i<=r) bs+=1;
    e(MPI_Recv(&(*scale)[index],bs,MPI_INT,i,TAG_SCALE_RESULT,comm,&status));
    index+=bs;
  }
}

/* Handle all errors returned from MPI_* function calls */
void e(int error) {
  if(error != MPI_SUCCESS) {
    fprintf(stderr,"Error starting MPI program, Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD,error);
    MPI_Finalize();
    exit(1);
  }
}
