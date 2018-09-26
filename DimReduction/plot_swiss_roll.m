function plot_swiss_roll(data, n, org)
    x = data(:,1);
    y = data(:,2);
    z = data(:,3);

    C = 1:n;
    C(x >= org & z > org) = 1;
    C(x < org & z >= org) = 2;
    C(x <= org & z < org) = 3;
    C(x > org & z <= org) = 4;

    figure
    scatter3(x,y,z,36,C,'filled')
    rotate3d on

end