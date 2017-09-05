function y = vl_nnmaxout_matlab(x, units, pieces, varargin)
% VL_MAXOUT_MATLAB maxout unit implemention in matlab
%   NOTE: While the VL_NNMAXOUT function (implemented in CUDA) is considerably 
%   faster this implementation exists as a reference for understanding the 
%   maxout operation.
%
% The maxout unit is described in : 
%    Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, 
%    A., & Bengio, Y. (2013). Maxout networks. arXiv preprint arXiv:1302.4389.
%
%    Y =  VL_MAXOUT(X, UNITS, PIECES) applies the maxout unit to the data X. 
%
%    DZDX =  vl_maxout(X, UNITS, PIECES, DZDY) computes the network derivative 
%    DZDX with respect to the input X given DZDY. DZDX has the same dimension 
%    as X.
%
% Copyright (C) 2017 Jia-Ren Chang, Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;
  sz = size(x) ;

 if isempty(dzdy)
   msg = 'units * pieces must match the number of input channels' ;
   assert(sz(3) == units * pieces, msg) ;
   y = zeros(sz(1), sz(2), units, size(x, 4), 'like', x) ;
   for i = 1:units
     y(:,:,i,:) = max(x(:,:, (i-1)*pieces +1 : i*pieces,:),[],3); 
   end
 else
   y = zeros(size(x), 'like', x) ;
   for i = 1:units 
     seq = (i-1)*pieces + 1 : i * pieces ;
     L = max(x(:,:,seq,:),[],3) ;
     mask = bsxfun(@eq,x(:,:,seq,:),L) ;  
     y(:,:,seq,:) = bsxfun(@times,mask,dzdy{1}(:,:,i,:)) ;
   end  
 end
