function neighbors = k_neighbors(points, point, k)
    n = size(points,1);
    points_sub = bsxfun(@minus, points, point);
    points_rows = mat2cell(points_sub, ones(1,n));
    dist = cellfun(@norm, points_rows);
    neighbors = get_n_smallest(dist, k+1);
    neighbors = neighbors(2:k+1);
end