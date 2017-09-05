type = 'double' ;
p = 7 ; u = 7 ; b = 3 ;
x = randn(1,1, p*u, b, type) ;
dzdy = randn(1,1,u, b, type) ;

y_cpu = vl_nnmaxout_matlab(x, u, p, dzdy) ;

x = gpuArray(x) ; dzdy = gpuArray(dzdy) ;
y_gpu = gather(vl_nnmaxout(x, u, p, dzdy)) ;

fprintf('-----------\n') ;
diff = y_gpu(:) - y_cpu(:) ;

fprintf('y_gpu: \n') ; 
disp(squeeze(y_gpu)) ;

fprintf('y_cpu: \n') ; 
disp(squeeze(y_cpu)) ;

fprintf('diff: %g\n', squeeze(diff)) ;
