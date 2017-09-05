classdef nnmaxout < nntest
  methods (Test)

    function basic(test)
      units = 5 ; pieces = 3 ; bs = 5 ;
      sz = [5, 8, units * pieces, bs] ;
      x = test.randn(sz) ;
      y = vl_nnmaxout(x, units, pieces) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnmaxout(x, units, pieces, dzdy) ;
      test.der(@(x) vl_nnmaxout(x, units, pieces), ...
                              x, dzdy, dzdx, 1e-4*test.range) ;
    end
  end
end
