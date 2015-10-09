      program cusparse_fortran_example
      use openacc
      implicit none
      external cuda_free
      integer cuda_memcpy_c2fort_real
      integer cusparse_create 
      external cusparse_destroy
      integer cusparse_get_version 
      integer cusparse_create_mat_descr
      external cusparse_destroy_mat_descr
      integer cusparse_set_mat_type 
      integer cusparse_get_mat_type
      integer cusparse_set_mat_index_base
      integer cusparse_get_mat_index_base
      integer cusparse_dcsrmv
      integer*8 handle
      integer*8 descrA      
      integer status
      integer cudaStat1,cudaStat2,cudaStat3
      integer cudaStat4,cudaStat5,cudaStat6
      integer n, nnz, nnz_vector
      parameter (n=4, nnz=9, nnz_vector=3)
      integer cooColIndexHostPtr(nnz)    
      real*8  cooValHostPtr(nnz)
      real*8  yHostPtr(2*n)
      integer csrRowHostPtr(n+1) 
      integer i, j
      integer version, mtype, ibase
      real*8  dtwo,dthree
c
      integer*8, allocatable, dimension(:) :: csrRowPtr
      integer*8, allocatable, dimension(:) :: cooColIndex
      real*8, allocatable, dimension(:) :: cooVal,y,ynp1
c
      write(*,*) "testing fortran example"
c
c     predefined constants (need to be careful with them)
      dtwo  = 2.0
      dthree= 3.0
c     create the following sparse test matrix in COO format 
c     (notice one-based indexing)
c     |1.0     2.0 3.0|
c     |    4.0        |
c     |5.0     6.0 7.0|
c     |    8.0     9.0| 
      cooColIndexHostPtr(1)=1 
      cooValHostPtr(1)     =1.0  
      cooColIndexHostPtr(2)=3 
      cooValHostPtr(2)     =2.0  
      cooColIndexHostPtr(3)=4 
      cooValHostPtr(3)     =3.0  
      cooColIndexHostPtr(4)=2 
      cooValHostPtr(4)     =4.0  
      cooColIndexHostPtr(5)=1 
      cooValHostPtr(5)     =5.0  
      cooColIndexHostPtr(6)=3 
      cooValHostPtr(6)     =6.0
      cooColIndexHostPtr(7)=4 
      cooValHostPtr(7)     =7.0  
      cooColIndexHostPtr(8)=2 
      cooValHostPtr(8)     =8.0  
      cooColIndexHostPtr(9)=4 
      cooValHostPtr(9)     =9.0  
c     print the matrix
      write(*,*) "Input data:"
      do i=1,nnz 
         write(*,*) "cooColIndexHostPtr[",i,"]=",cooColIndexHostPtr(i)
         write(*,*) "cooValHostPtr[",     i,"]=",cooValHostPtr(i)
      enddo
  
c     create a sparse and dense vector  
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
      csrRowHostPtr(1)=0
      csrRowHostPtr(2)=3
      csrRowHostPtr(3)=4 
      csrRowHostPtr(4)=7 
      csrRowHostPtr(5)=9 
    
c     print the vectors
      do j=1,2
         do i=1,n        
            write(*,*) "yHostPtr[",i,",",j,"]=",yHostPtr(i+n*(j-1))
         enddo
      enddo
      allocate(cooColIndex(nnz),cooVal(nnz),
     &         y(2*n),csrRowPtr(n+1),ynp1(2*n))
!$acc data create(cooColIndex,cooVal,y,csrRowPtr,ynp1)
!$acc kernels
      cooColIndex(:)=cooColIndexHostPtr(:)
      cooVal(:)=cooValHostPtr(:)
      y(:)=yHostPtr(:)
      csrRowPtr(:)=csrRowHostPtr(:)
      ynp1(:)=0.d0
!$acc end kernels
c
c     initialize cusparse library
c     CUSPARSE_STATUS_SUCCESS=0 
      status = cusparse_create(handle)
      if (status /= 0) then 
         write(*,*) "CUSPARSE Library initialization failed"
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(y)  
         call cuda_free(csrRowPtr)  
         call cuda_free(ynp1)  
         stop 1
      endif
c     get version
c     CUSPARSE_STATUS_SUCCESS=0
      status = cusparse_get_version(handle,version)
      if (status /= 0) then 
         write(*,*) "CUSPARSE Library initialization failed"
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(y)   
         call cuda_free(csrRowPtr)  
         call cuda_free(ynp1)  
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
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(y)  
         call cuda_free(csrRowPtr)  
         call cuda_free(ynp1)  
         call cusparse_destroy(handle)
         stop 1
      endif  
      status = cusparse_set_mat_type(descrA,0)       
      status = cusparse_set_mat_index_base(descrA,1) 
c     print the matrix descriptor
      mtype = cusparse_get_mat_type(descrA)
      ibase = cusparse_get_mat_index_base(descrA) 
      write (*,*) "matrix descriptor:"
      write (*,*) "t=",mtype,"b=",ibase
c
c     exercise Level 2 routines (csrmv) 
c     CUSPARSE_STATUS_SUCCESS=0
c     CUSPARSE_OPERATION_NON_TRANSPOSE=0
      print*,'1:Am I HERE?'
!$acc host_data use_device(cooVal,csrRowPtr,cooColIndex,y,ynp1)
      status= cusparse_dcsrmv(handle, 0, n, n, nnz, dtwo,
     $                       descrA, cooVal, csrRowPtr, cooColIndex, 
     $                       y, dthree, ynp1)
!$acc end host_data     
      print*,'2:Can I get HERE?',status
      if (status /= 0) then 
         call cuda_free(cooColIndex)    
         call cuda_free(cooVal)
         call cuda_free(y)  
         call cuda_free(csrRowPtr)  
         call cuda_free(ynp1)  
         call cusparse_destroy_mat_descr(descrA)
         call cusparse_destroy(handle)
         write(*,*) "Matrix-vector multiplication failed"
         stop 1
      endif    
c   
c     print intermediate results (y) 
c     y = [10 20 30 40 | 680 760 1230 2240]
c     cudaSuccess=0
c     cudaMemcpyDeviceToHost=2
c      cudaStat1 = cuda_memcpy_c2fort_real(yHostPtr, y, 2*n*8, 2) 
c      if (cudaStat1 /= 0) then  
c         call cuda_free(cooColIndex)    
c         call cuda_free(cooVal)
c         call cuda_free(y)  
c         call cuda_free(csrRowPtr)
c         call cusparse_destroy_mat_descr(descrA)
c         call cusparse_destroy(handle)
c         write(*,*) "Memcpy from Device to Host failed"
c         stop 1
c      endif
!$acc end data
      yHostPtr(:)=y(:)
c 
      write(*,*) "Final results:"
      do j=1,2
         do i=1,n        
             write(*,*) "yHostPtr[",i,",",j,"]=",yHostPtr(i+n*(j-1))
         enddo
      enddo 

c     deallocate GPU memory and exit      
      call cuda_free(cooColIndex)    
      call cuda_free(cooVal)
      call cuda_free(y)  
      call cuda_free(csrRowPtr)
      call cuda_free(ynp1)  
      call cusparse_destroy_mat_descr(descrA)
      call cusparse_destroy(handle)

      stop 0
      end
