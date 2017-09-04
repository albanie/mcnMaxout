function y = vl_nnmaxout(x, units, pieces, varargin)
% VL_MAXOUT maxout unit implemention
%
% The maxout unit is described in : 
%    Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, 
%    A., & Bengio, Y. (2013). Maxout networks. arXiv preprint arXiv:1302.4389.
%
%    Y =  vl_maxout(X, UNITS, PIECES) applies the maxout unit to the data X. 
%
%    DZDX =  vl_maxout(X, UNITS, PIECES, DZDY) computes the network derivative 
%    DZDX with respect to the input X given DZDY. 
%    DZDX has the same dimension as X.
%
%   based on the implementation by Jia Ren Chang (NCTU, Taiwan)

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;
  sz = size(x) ;

  if isempty(dzdy)
   msg = 'units * pieces must match the number of input channels' ;
   assert(sz(3) == units * pieces, msg) ;
   y = zeros(sz(1), sz(2), units, sz(4), 'like', x) ;
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
