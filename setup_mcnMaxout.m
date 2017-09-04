function setup_mcnMaxout()
%SETUP_MCNMAXOUT Sets up mcnMaxout, by adding its folders to the Matlab path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab']) ;
  addpath([root '/example']) ;
