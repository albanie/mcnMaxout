function net = cnn_init_maxout
% CNN_INIT_MAXOUT initialise a small maxout network
%   NET = CNN_INIT_MAXOUT constructs a small classification network with 
%   maxout units
% 
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  data = Input('data') ; labels = Input('label') ;
  u = [192 160 96] ; p = [1 5 5] ; ker = [5 5] ; poolKer = [3 3] ; pad = 2 ; 
  m1 = add_maxout_block(data, u, p, ker, pad, poolKer) ;
  u = [192 192 192] ; p = [1 5 5] ; ker = [5 5] ; poolKer = [3 3] ; pad = 2 ;
  m2 = add_maxout_block(m1, u, p, ker, pad, poolKer) ;
  u = [192 192 10] ; p = [1 5 5] ; ker = [3 3] ; poolKer = [8 8] ; pad = 1 ;
  m3 = add_maxout_block(m2, u, p, ker, pad, poolKer) ;
  lopts = {'numInputDer', 1} ;
  loss = Layer.create(@vl_nnloss, {m3, labels}, lopts{:}) ;
  cls_err = Layer.create(@vl_nnloss, {m3, labels, 'loss', 'classerror'}, lopts{:}) ;
  loss.name = 'logloss' ; cls_err.name = 'classerror' ;
  net = Net(loss, cls_err) ; net.meta.inputSize = [32 32 3] ;

% ------------------------------------------------------------------
function block = add_maxout_block(in, u, p, ker, pad, poolKer) 
% ------------------------------------------------------------------
  c1 = add_block(in, [ker(1:2), 3, u(1)*p(1)], 1, 'stride', 1, 'pad', pad) ;
  c2 = add_block(c1, [1, 1, u(1)*p(1), u(2)*p(2)], 1, 'stride', 1, 'pad', 0) ;
  m1 = Layer.create(@vl_nnmaxout, {c2, u(2), p(2)}) ;
  c3 = add_block(m1, [1, 1, u(2), u(3)*p(3)], 1, 'stride', 1, 'pad', 0) ;
  m2 = Layer.create(@vl_nnmaxout, {c3, u(3), p(3)}) ;
  p1 = vl_nnpool(m2, poolKer, 'method', 'avg', 'stride', 2, 'pad', [0 1 0 1]) ;
  block = vl_nndropout(p1, 'rate', 0.5) ;

% ---------------------------------------------
function net = add_block(net, sz, bn, varargin)
% ---------------------------------------------
  filters = Param('value', orthoInit(sz, 'single'), 'learningRate', 1) ;
  biases = Param('value', zeros(sz(4), 1, 'single'), 'learningRate', 2) ;
  net = vl_nnconv(net, filters, biases, varargin{:}) ;
  if bn
    net = vl_nnbnorm(net, 'learningRate', [2 1 0.05], 'testMode', false) ;
  end

% ------------------------------
function x = orthoInit(sz, type)
% ------------------------------
% ORTHOINIT(SZ, TYPE) - orthoInital weight initialisation
%   X = ORTHOINIT(SZ) initialises a tensor of weights X
%   with dimensions SZ such that the fourth dimension of X
%   is orthonormal to the others
%
% NOTES:
%   This form of initialisiation is introduced in the paper:
%   "Exact solutions to the nonlinear dynamics of learning in 
%   deep linear neural networks", Saxe et. al, 2013
%   https://arxiv.org/abs/1312.6120

    initw = randn(prod(sz(1:3)), sz(4), type) ;
    [U,~,V] = svd(initw,'econ') ;
    if numel(V) == prod(sz), src = V ; else, src = U ; end
    x = reshape(src, sz(1), sz(2), sz(3), sz(4)) ;
