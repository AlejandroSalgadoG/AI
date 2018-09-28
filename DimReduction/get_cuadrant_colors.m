function colors = get_cuadrant_colors(x, z)
    n = size(x,1);
    colors = zeros(n,1);
    
    m = 0.85;
    pt = [12.5 0];
    
    % cuadrant 1
    colors(x >= 0 & z > 0) = 1;    
    colors(z >= -m*(x-pt(1))) = 2;
    % cuadrant 2
    colors(x > 0 & z <= 0) = 3;
    colors(z < m*(x-pt(1))) = 4;
    % cuadrant 4
    colors(x <= 0 & z < 0 ) = 5;
end