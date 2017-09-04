p = 2 ; u = 2 ;
x = randn(1,1, p*u, 1, 'single') ;

y_cpu = vl_nnmaxout_matlab(x, p, u) ;

x = gpuArray(x) ;
y_gpu = gather(vl_nnmaxout(x, p, u, 'Verbose')) ;

fprintf('-----------\n') ;
diff = y_gpu(:) - y_cpu(:) ;

fprintf('y_gpu: \n') ; 
disp(squeeze(y_gpu)) ;

fprintf('y_cpu: \n') ; 
disp(squeeze(y_cpu)) ;

fprintf('diff: %g\n', squeeze(diff)) ;
