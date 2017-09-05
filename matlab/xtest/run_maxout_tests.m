function run_maxout_tests(varargin)
% ------------------------
% run tests for mcnMaxout module
% (based on vl_testnn)
% ------------------------

opts.cpu = false ; % not currently supported
opts.gpu = true ;
opts.single = true ;
opts.double = false ; % current cuda impl uses atomicAdd
opts.command = 'nn' ;
opts = vl_argparse(opts, varargin) ;

import matlab.unittest.constraints.* ;
import matlab.unittest.selectors.* ;
import matlab.unittest.plugins.TAPPlugin;
import matlab.unittest.plugins.ToFile;

% pick tests
sel = HasName(StartsWithSubstring(opts.command)) ;
if ~opts.gpu
  sel = sel & ~HasName(ContainsSubstring('device=gpu')) ;
end
if ~opts.cpu
  sel = sel & ~HasName(ContainsSubstring('device=cpu')) ;
end
if ~opts.double
  sel = sel & ~HasName(ContainsSubstring('dataType=double')) ;
end
if ~opts.single
  sel = sel & ~HasName(ContainsSubstring('dataType=single')) ;
end

% add test class to path
suiteDir = fullfile(vl_rootnn, 'contrib', 'mcnMaxout/matlab/xtest/suite') ;
addpath(suiteDir) ;
suite = matlab.unittest.TestSuite.fromFolder(suiteDir, sel) ;
runner = matlab.unittest.TestRunner.withTextOutput('Verbosity',3);
result = runner.run(suite);
display(result)
