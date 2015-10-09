FC=pgfortran
NVCC=nvcc

FFLAGS= -O3 -mp -acc -ta=nvidia -Minfo=accel -c
CFLAGS= -O3 -c
LINK= -L/usr/local/cuda-6.5/lib64 -lcudart -lcusparse -acc

all: cuSPARSE 

cuSPARSE: cusparse_fortran.o cuSPARSE_DCSRMV.o
	$(FC) cusparse_fortran.o cuSPARSE_DCSRMV.o -o cuSPARSE $(LINK)

cusparse_fortran.o: cusparse_fortran.c
	$(NVCC) $(CFLAGS) cusparse_fortran.c
	
cuSPARSE_DCSRMV.o: cuSPARSE_DCSRMV.f
	$(FC) $(FFLAGS) cuSPARSE_DCSRMV.f

clean:
	rm *o cuSPARSE

#pgfortran -Mcuda -o cuSPARSE_DCSRMV cuSPARSE_DCSRMV.o cusparse_fortran.o -L/usr/local/cuda-6.5/lib64 -lcudart -lcusparse
#pgfortran -mp -O3 -acc -ta=nvidia -Minfo=accel -c cuSPARSE_DCSRMV.f
