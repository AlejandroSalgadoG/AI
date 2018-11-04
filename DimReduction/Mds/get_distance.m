function D = get_distance(points)
    n = size(points, 1);
    D = zeros(n);
    for j = 1:n
        D(:,j) = sqrt(sum(bsxfun(@minus, points, points(j,:)).^2, 2));
    end
end