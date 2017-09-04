function [net, info] = cnn_cifar_maxout(varargin)
% CNN_CIFAR_MAXOUT train a simple maxout network on cifar
%   [NET, INFO] = CNN_CIFAR_MAXOUT(VARARGIN) provides a simple example
%   of training a maxout network on CIFAR.  

%   This example is based on the `cnn_cifar.m` script provided in MatConvNet.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.gpus = 3 ;
  opts.continue = 1 ;
  opts.whitenData = true ;
  opts.modelName = 'maxout' ;
  opts.contrastNormalization = true ;
  opts.dataDir = fullfile(vl_rootnn, 'data','cifar') ;
  opts.expDir = fullfile(vl_rootnn, 'data', sprintf('cifar-%s', opts.modelName)) ;
  opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
  opts.benchmark = false ; 
  opts = vl_argparse(opts, varargin) ;

  % set learning rate
  steady = 0.5 * ones(1,80) ;
  gentle = 0.5:-0.005:0.005 ;
  vGentle = 0.005:-0.0001:0.00005 ;
  opts.train.learningRate = [steady gentle vGentle] ;

  opts.train.gpus = opts.gpus ;
  opts.train.batchSize = 100 ;
  opts.train.numEpochs = 222;
  opts.train.weightDecay = 0.0005 ;
  opts.train.continue = opts.continue ;

  net = cnn_init_maxout ;

  if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
  else
    imdb = getCifarImdb(opts) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
  end

  if opts.benchmark % dev-only
    sample = 1000 ;
    imdb.images.set = (imdb.images.set * 0) + 4 ;
    imdb.images.set(1:sample) = 1 ;
    imdb.images.set(sample+1:2 * sample) = 2 ;
    opts.train.numEpochs = 1 ;
  end

  opts.train.val = find(imdb.images.set == 3) ;
  net.meta.classes.name = imdb.meta.classes(:)' ;
  [net, info] = cnn_train_autonn(net, imdb, ...
                     @(i,b) getBatch(i, b, opts), ...
                     opts.train, 'expDir', opts.expDir) ;

% -------------------------------------------------------------------------
function inputs = getBatch(imdb, batch, opts)
% -------------------------------------------------------------------------
  images = imdb.images.data(:,:,:,batch) ;
  labels = imdb.images.labels(1,batch) ;
  if rand > 0.5, images=fliplr(images) ; end
  if numel(opts.train.gpus) > 0, images = gpuArray(images) ; end
  inputs = {'data', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
  % Preapre the imdb structure, returns image data with mean image subtracted
  unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
  files = [arrayfun(@(n) {sprintf('data_batch_%d.mat', n)}, 1:5) ...
    {'test_batch.mat'}];
  files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
  file_set = uint8([ones(1, 5), 3]);

  if any(cellfun(@(fn) ~exist(fn, 'file'), files))
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
    fprintf('downloading %s\n', url) ;
    untar(url, opts.dataDir) ;
  end

  data = cell(1, numel(files));
  labels = cell(1, numel(files));
  sets = cell(1, numel(files));
  for fi = 1:numel(files)
    fd = load(files{fi}) ;
    data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
    labels{fi} = fd.labels' + 1; % Index from 1
    sets{fi} = repmat(file_set(fi), size(labels{fi}));
  end

  set = cat(2, sets{:});
  data = single(cat(4, data{:}));

  % remove mean in any case
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  % normalize by image mean and std as suggested in `An Analysis of
  % Single-Layer Networks in Unsupervised Feature Learning` Adam
  % Coates, Honglak Lee, Andrew Y. Ng

  if opts.contrastNormalization
    z = reshape(data,[],60000) ;
    z = bsxfun(@minus, z, mean(z,1)) ;
    n = std(z,0,1) ;
    z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
    data = reshape(z, 32, 32, 3, []) ;
  end

  if opts.whitenData
    z = reshape(data,[],60000) ;
    W = z(:,set == 1)*z(:,set == 1)'/60000 ;
    [V,D] = eig(W) ;
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D) ;
    en = sqrt(mean(d2)) ;
    z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
    data = reshape(z, 32, 32, 3, []) ;
  end

  clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

  imdb.images.data = data ;
  imdb.images.labels = single(cat(2, labels{:})) ;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'} ;
  imdb.meta.classes = clNames.label_names;
