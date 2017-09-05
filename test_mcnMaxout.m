function test_mcnMaxout
% run tests for mcnMaxout module
%
% Copyright (C) 2017 Jia-Ren Chang, Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

% add tests to path
addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab/xtest')) ;
addpath(fullfile(vl_rootnn, 'matlab/xtest/suite')) ;

% test network layers
run_maxout_tests('command', 'nn') ;
