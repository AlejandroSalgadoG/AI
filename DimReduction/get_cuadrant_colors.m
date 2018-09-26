function colors = get_cuadrant_colors(x, z, org)
    n = size(x,1);
    colors = 1:n;
    colors(x >= org & z > org) = 1;
    colors(x < org & z >= org) = 2;
    colors(x <= org & z < org) = 3;
    colors(x > org & z <= org) = 4;
end