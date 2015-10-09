      program cusparse_fortran_example
      implicit none
      integer cuda_malloc
      external cuda_free
      integer cuda_memcpy_c2fort_int
      integer cuda_memcpy_c2fort_real
      integer cuda_memcpy_fort2c_int
      integer cuda_memcpy_fort2c_real
      integer cuda_memset
      integer cusparse_create 
      external cusparse_destroy
      integer cusparse_get_version 
      integer cusparse_create_mat_descr
      external cusparse_destroy_mat_descr
      integer cusparse_set_mat_type 
      integer cusparse_get_mat_type
      integer cusparse_get_mat_fill_mode
      integer cusparse_get_mat_diag_type
      integer cusparse_set_mat_index_base
      integer cusparse_get_mat_index_base
      integer cusparse_xcoo2csr
      integer cusparse_dsctr
      integer cusparse_dcsrmv
      integer cusparse_dcsrmm
      external get_shifted_address
      integer*8 handle
      integer*8 descrA      
      integer*8 cooRowIndex
      integer*8 cooColIndex    
      integer*8 cooVal
      integer*8 xInd
      integer*8 xVal
      integer*8 y  
      integer*8 z 
      integer*8 csrRowPtr
      integer*8 ynp1  
      integer status
      integer cudaStat1,cudaStat2,cudaStat3
      integer cudaStat4,cudaStat5,cudaStat6
      integer n, nnz, nnz_vector
      parameter (n=4, nnz=9, nnz_vector=3)
      integer cooRowIndexHostPtr(nnz)
      integer cooColIndexHostPtr(nnz)    
      real*8  cooValHostPtr(nnz)
      integer xIndHostPtr(nnz_vector)
      real*8  xValHostPtr(nnz_vector)
      real*8  yHostPtr(2*n)
      real*8  zHostPtr(2*(n+1)) 
      integer i, j
      integer version, mtype, fmode, dtype, ibase
      real*8  dzero,dtwo,dthree,dfive
      real*8  epsilon


      write(*,*) "testing fortran example"

c     predefined constants (need to be careful with them)
      dzero = 0.0
      dtwo  = 2.0
      dthree= 3.0
      dfive = 5.0
c     create the following sparse test matrix in COO format 
c     (notice one-based indexing)
c     |1.0     2.0 3.0|
c     |    4.0        |
c     |5.0     6.0 7.0|
c     |    8.0     9.0| 
      cooRowIndexHostPtr(1)=1 
      cooColIndexHostPtr(1)=1 
      cooValHostPtr(1)     =1.0  
      cooRowIndexHostPtr(2)=1 
      cooColIndexHostPtr(2)=3 
      cooValHostPtr(2)     =2.0  
      cooRowIndexHostPtr(3)=1 
      cooColIndexHostPtr(3)=4 
      cooValHostPtr(3)     =3.0  
      cooRowIndexHostPtr(4)=2 
      cooColIndexHostPtr(4)=2 
      cooValHostPtr(4)     =4.0  
      cooRowIndexHostPtr(5)=3 
      cooColIndexHostPtr(5)=1 
      cooValHostPtr(5)     =5.0  
      cooRowIndexHostPtr(6)=3 
      cooColIndexHostPtr(6)=3 
      cooValHostPtr(6)     =6.0
      cooRowIndexHostPtr(7)=3 
      cooColIndexHostPtr(7)=4 
      cooValHostPtr(7)     =7.0  
      cooRowIndexHostPtr(8)=4 
      cooColIndexHostPtr(8)=2 
      cooValHostPtr(8)     =8.0  
      cooRowIndexHostPtr(9)=4 
      cooColIndexHostPtr(9)=4 
      cooValHostPtr(9)     =9.0  
c     print the matrix
      write(*,*) "Input data:"
      do i=1,nnz        
         write(*,*) "cooRowIndexHostPtr[",i,"]=",cooRowIndexHostPtr(i)
         write(*,*) "cooColIndexHostPtr[",i,"]=",cooColIndexHostPtr(i)
         write(*,*) "cooValHostPtr[",     i,"]=",cooValHostPtr(i)
      enddo
  
c     create a sparse and dense vector  
c     xVal= [100.0 200.0 400.0]   (sparse)
c     xInd= [0     1     3    ]
c     y   = [10.0 20.0 30.0 40.0 | 50.0 60.0 70.0 80.0] (dense) 
c     (notice one-based indexing)
      yHostPtr(1) = 10.0  
      yHostPtr(2) = 20.0  
      yHostPtr(3) = 30.0
      yHostPtr(4) = 40.0  
      yHostPtr(5) = 50.0
      yHostPtr(6) = 60.0
      yHostPtr(7) = 70.0
      yHostPtr(8) = 80.0
      xIndHostPtr(1)=1 
      xValHostPtr(1)=100.0 
      xIndHostPtr(2)=2 
      xValHostPtr(2)=200.0
      xIndHostPtr(3)=4 
      xValHostPtr(3)=400.0    
c     print the vectors
      do j=1,2
         do i=1,n        
            write(*,*) "yHostPtr[",i,",",j,"]=",yHostPtr(i+n*(j-1))
         enddo
      enddo
      do i=1,nnz_vector        
         write(*,*) "xIndHostPtr[",i,"]=",xIndHostPtr(i)
         write(*,*) "xValHostPtr[",i,"]=",xValHostPtr(i)
      enddo

c     allocate GPU memory and copy the matrix and vectors into it 
c     cudaSuccess=0
c     cudaMemcpyHostToDevice=1
      cudaStat1 = cuda_malloc(cooRowIndex,nnz*4) 
      cudaStat2 = cuda_malloc(cooColIndex,nnz*4)
      cudaStat3 = cuda_malloc(cooVal,     nnz*8) 
      cudaStat4 = cuda_malloc(y,          2*n*8)   
      cudaStat5 = cuda_malloc(xInd,nnz_vector*4) 
      cudaStat6 = cuda_malloc(xVal,nnz_vector*8) 
      if ((cudaStat1 /= 0) .OR. 
     $    (cudaStat2 /= 0) .OR. 
     $    (cudaStat3 /= 0) .OR. 
     $    (cudaStat4 /= 0) .OR. 
     $    (cudaStat5 /= 0) .OR. 
     $    (cudaStat6 /= 0)) then 
         write(*,*) "Device malloc failed"
         write(*,*) "cudaStat1=",cudaStat1
         write(*,*) "cudaStat2=",cudaStat2
         write(*,*) "cudaStat3=",cudaStat3
         write(*,*) "cudaStat4=",cudaStat4
         write(*,*) "cudaStat5=",cudaStat5
         write(*,*) "cudaStat6=",cudaStat6
         stop 2 
      endif    
      cudaStat1 = cuda_memcpy_fort2c_int(cooRowIndex,cooRowIndexHostPtr, 
     $                                   nnz*4,1)        
      cudaStat2 = cuda_memcpy_fort2c_int(cooColIndex,cooColIndexHostPtr, 
     $                                   nnz*4,1)       
      cudaStat3 = cuda_memcpy_fort2c_real(cooVal,    cooValHostPtr,      
     $                                    nnz*8,1)       
      cudaStat4 = cuda_memcpy_fort2c_real(y,      yHostPtr,           
     $                                    2*n*8,1)       
      cudaStat5 = cuda_memcpy_fort2c_int(xInd,       xIndHostPtr,        
     $                                   nnz_vector*4,1) 
      cudaStat6 = cuda_memcpy_fort2c_real(xVal,      xValHostPtr,        
     $                                    nnz_vector*8,1)
      if ((cudaStat1 /= 0) .OR. 
     $    (cudaStat2 /= 0) .OR. 
     $    (cudaStat3 /= 0) .OR. 
     $    (cudaStat4 /= 0) .OR. 
     $    (cudaStat5 /= 0) .OR. 
     $    (cudaStat6 /= 0)) then 
         write(*,*) "Memcpy from Host to Device failed"
         write(*,*) "cudaStat1=",cudaStat1
         write(*,*) "cudaStat2=",cudaStat2
         write(*,*) "cudaStat3=",cudaStat3
         write(*,*) "cudaStat4=",cudaStat4
         write(*,*) "cudaStat5=",cudaStat5
         write(*,*) "cudaStat6=",cudaStat6
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         stop 1
      endif
    
c     initialize cusparse library
c     CUSPARSE_STATUS_SUCCESS=0 
      status = cusparse_create(handle)
      if (status /= 0) then 
         write(*,*) "CUSPARSE Library initialization failed"
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         stop 1
      endif
c     get version
c     CUSPARSE_STATUS_SUCCESS=0
      status = cusparse_get_version(handle,version)
      if (status /= 0) then 
         write(*,*) "CUSPARSE Library initialization failed"
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)   
         call cusparse_destroy(handle)
         stop 1
      endif
      write(*,*) "CUSPARSE Library version",version

c     create and setup the matrix descriptor
c     CUSPARSE_STATUS_SUCCESS=0 
c     CUSPARSE_MATRIX_TYPE_GENERAL=0
c     CUSPARSE_INDEX_BASE_ONE=1  
      status= cusparse_create_mat_descr(descrA) 
      if (status /= 0) then 
         write(*,*) "Creating matrix descriptor failed"
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         call cusparse_destroy(handle)
         stop 1
      endif  
      status = cusparse_set_mat_type(descrA,0)       
      status = cusparse_set_mat_index_base(descrA,1) 
c     print the matrix descriptor
      mtype = cusparse_get_mat_type(descrA)
      fmode = cusparse_get_mat_fill_mode(descrA) 
      dtype = cusparse_get_mat_diag_type(descrA) 
      ibase = cusparse_get_mat_index_base(descrA) 
      write (*,*) "matrix descriptor:"
      write (*,*) "t=",mtype,"m=",fmode,"d=",dtype,"b=",ibase

c     exercise conversion routines (convert matrix from COO 2 CSR format) 
c     cudaSuccess=0
c     CUSPARSE_STATUS_SUCCESS=0 
c     CUSPARSE_INDEX_BASE_ONE=1
      cudaStat1 = cuda_malloc(csrRowPtr,(n+1)*4)
      if (cudaStat1 /= 0) then  
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         call cusparse_destroy_mat_descr(descrA)
         call cusparse_destroy(handle)
         write(*,*) "Device malloc failed (csrRowPtr)"
         stop 2
      endif
      status= cusparse_xcoo2csr(handle,cooRowIndex,nnz,n,
     $                          csrRowPtr,1)         
      if (status /= 0) then 
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         call cuda_free(csrRowPtr)
         call cusparse_destroy_mat_descr(descrA)
         call cusparse_destroy(handle)
         write(*,*) "Conversion from COO to CSR format failed"
         stop 1
      endif  
c     csrRowPtr = [0 3 4 7 9] 

c     exercise Level 1 routines (scatter vector elements)
c     CUSPARSE_STATUS_SUCCESS=0  
c     CUSPARSE_INDEX_BASE_ONE=1
      call get_shifted_address(y,n*8,ynp1)
      status= cusparse_dsctr(handle, nnz_vector, xVal, xInd, 
     $                       ynp1, 1)
      if (status /= 0) then 
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         call cuda_free(csrRowPtr)
         call cusparse_destroy_mat_descr(descrA)
         call cusparse_destroy(handle)
         write(*,*) "Scatter from sparse to dense vector failed"
         stop 1
      endif  
c     y = [10 20 30 40 | 100 200 70 400]

c     exercise Level 2 routines (csrmv) 
c     CUSPARSE_STATUS_SUCCESS=0
c     CUSPARSE_OPERATION_NON_TRANSPOSE=0
      status= cusparse_dcsrmv(handle, 0, n, n, nnz, dtwo,
     $                       descrA, cooVal, csrRowPtr, cooColIndex, 
     $                       y, dthree, ynp1)        
      if (status /= 0) then 
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         call cuda_free(csrRowPtr)
         call cusparse_destroy_mat_descr(descrA)
         call cusparse_destroy(handle)
         write(*,*) "Matrix-vector multiplication failed"
         stop 1
      endif    
    
c     print intermediate results (y) 
c     y = [10 20 30 40 | 680 760 1230 2240]
c     cudaSuccess=0
c     cudaMemcpyDeviceToHost=2
      cudaStat1 = cuda_memcpy_c2fort_real(yHostPtr, y, 2*n*8, 2) 
      if (cudaStat1 /= 0) then  
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         call cuda_free(csrRowPtr)
         call cusparse_destroy_mat_descr(descrA)
         call cusparse_destroy(handle)
         write(*,*) "Memcpy from Device to Host failed"
         stop 1
      endif
      write(*,*) "Intermediate results:"
      do j=1,2
         do i=1,n        
             write(*,*) "yHostPtr[",i,",",j,"]=",yHostPtr(i+n*(j-1))
         enddo
      enddo 

c     exercise Level 3 routines (csrmm)
c     cudaSuccess=0 
c     CUSPARSE_STATUS_SUCCESS=0
c     CUSPARSE_OPERATION_NON_TRANSPOSE=0
      cudaStat1 = cuda_malloc(z, 2*(n+1)*8)   
      if (cudaStat1 /= 0) then  
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         call cuda_free(csrRowPtr)
         call cusparse_destroy_mat_descr(descrA)
         call cusparse_destroy(handle)
         write(*,*) "Device malloc failed (z)"
         stop 2
      endif
      cudaStat1 = cuda_memset(z, 0, 2*(n+1)*8)    
      if (cudaStat1 /= 0) then  
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         call cuda_free(z) 
         call cuda_free(csrRowPtr)
         call cusparse_destroy_mat_descr(descrA)
         call cusparse_destroy(handle)
         write(*,*) "Memset on Device failed"
         stop 1
      endif
      status= cusparse_dcsrmm(handle, 0, n, 2, n, nnz, dfive, 
     $                        descrA, cooVal, csrRowPtr, cooColIndex, 
     $                        y, n, dzero, z, n+1) 
      if (status /= 0) then     
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         call cuda_free(z) 
         call cuda_free(csrRowPtr)
         call cusparse_destroy_mat_descr(descrA)
         call cusparse_destroy(handle)
         write(*,*) "Matrix-matrix multiplication failed"
         stop 1
      endif  

c     print final results (z) 
c     cudaSuccess=0
c     cudaMemcpyDeviceToHost=2
      cudaStat1 = cuda_memcpy_c2fort_real(zHostPtr, z, 2*(n+1)*8, 2) 
      if (cudaStat1 /= 0) then 
         call cuda_free(cooRowIndex)
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(xInd)
         call cuda_free(xVal)
         call cuda_free(y)  
         call cuda_free(z) 
         call cuda_free(csrRowPtr)
         call cusparse_destroy_mat_descr(descrA)
         call cusparse_destroy(handle)
         write(*,*) "Memcpy from Device to Host failed"
         stop 1
      endif 
c     z = [950 400 2550 2600 0 | 49300 15200 132300 131200 0]
      write(*,*) "Final results:"
      do j=1,2
         do i=1,n+1
            write(*,*) "z[",i,",",j,"]=",zHostPtr(i+(n+1)*(j-1))
         enddo
      enddo
    
c     check the results 
      epsilon = 0.00000000000001
      if ((DABS(zHostPtr(1) - 950.0)   .GT. epsilon)  .OR. 
     $    (DABS(zHostPtr(2) - 400.0)   .GT. epsilon)  .OR.  
     $    (DABS(zHostPtr(3) - 2550.0)  .GT. epsilon)  .OR.  
     $    (DABS(zHostPtr(4) - 2600.0)  .GT. epsilon)  .OR.  
     $    (DABS(zHostPtr(5) - 0.0)     .GT. epsilon)  .OR.  
     $    (DABS(zHostPtr(6) - 49300.0) .GT. epsilon)  .OR.  
     $    (DABS(zHostPtr(7) - 15200.0) .GT. epsilon)  .OR.  
     $    (DABS(zHostPtr(8) - 132300.0).GT. epsilon)  .OR.  
     $    (DABS(zHostPtr(9) - 131200.0).GT. epsilon)  .OR.  
     $    (DABS(zHostPtr(10) - 0.0)    .GT. epsilon)  .OR. 
     $    (DABS(yHostPtr(1) - 10.0)    .GT. epsilon)  .OR.  
     $    (DABS(yHostPtr(2) - 20.0)    .GT. epsilon)  .OR.  
     $    (DABS(yHostPtr(3) - 30.0)    .GT. epsilon)  .OR.  
     $    (DABS(yHostPtr(4) - 40.0)    .GT. epsilon)  .OR.  
     $    (DABS(yHostPtr(5) - 680.0)   .GT. epsilon)  .OR.  
     $    (DABS(yHostPtr(6) - 760.0)   .GT. epsilon)  .OR.  
     $    (DABS(yHostPtr(7) - 1230.0)  .GT. epsilon)  .OR.  
     $    (DABS(yHostPtr(8) - 2240.0)  .GT. epsilon)) then 
          write(*,*) "fortran example test FAILED"
       else
          write(*,*) "fortran example test PASSED"
       endif   
       
c      deallocate GPU memory and exit      
       call cuda_free(cooRowIndex)
       call cuda_free(cooColIndex)    
       call cuda_free(cooVal)
       call cuda_free(xInd)
       call cuda_free(xVal)
       call cuda_free(y)  
       call cuda_free(z) 
       call cuda_free(csrRowPtr)
       call cusparse_destroy_mat_descr(descrA)
       call cusparse_destroy(handle)

       stop 0
       end
