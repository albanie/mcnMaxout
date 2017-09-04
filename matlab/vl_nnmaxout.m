% VL_MAXOUT maxout unit implemention
%
% The maxout unit is described in : 
%    Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, 
%    A., & Bengio, Y. (2013). Maxout networks. arXiv preprint arXiv:1302.4389.
%
%    Y =  VL_MAXOUT(X, UNITS, PIECES) applies the maxout unit to the data X. 
%
%    DZDX =  vl_maxout(X, UNITS, PIECES, DZDY) computes the network derivative 
%    DZDX with respect to the input X given DZDY. 
%    DZDX has the same dimension as X.
%
% Copyright (C) 2017 Jia-Ren Chang, Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]
