function colors = paint_neigbors(n, point, neighbors)
    colors = zeros(n,1);
    colors(point) = 1;
    colors(neighbors) = 2;
end