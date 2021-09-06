mat_name = 'd10';
load(sprintf('%s.mat',mat_name));

%% clouds 
%% left_imgs
%% right_imgs

% extrinsic = [1.4873383572288623e-02, -7.5262107631216768e-02, 9.9705285597906945e-01, 1.1726553446679751e-01;
%        -9.9972245421128814e-01, -1.9339637517893360e-02, 1.3453362644172184e-02, -7.6538407521982627e-02;
%        1.8270112393487914e-02, -9.9697622518071305e-01, -7.5528864797147044e-02, -5.3472149952639637e-02];
% 
%    ext = extrinsic(1:3, 1:3);
%    t = extrinsic(1:3,4);
%    new_ext = [ext' -ext'*t];
   
extrinsic = [-0.0077295, -0.9995, -0.030778, -0.075815; 
    -0.031705, 0.031008, -0.99902, -0.11444;
    0.99947, -0.0067461, -0.031928, -0.0041746];

new_ext = extrinsic;
   
intrinsic = [6.5717649994077067e+02, 0, 6.7283525424752838e+02;
        0, 6.5708440653270486e+02, 3.9887849408959755e+02;
        0, 0, 1 ];

for i = 1:size(clouds,2)
    img = left_imgs{i};
    [m,n] = size(img);
    imshow(img);
    hold on;

    cloud = clouds{i};
    X = cloud(:,1);
    Y = cloud(:,2);
    Z = cloud(:,3);
    velo_points = [X, Y, Z, ones(size(X,1), 1)];
    image_pixels = intrinsic*new_ext*velo_points';
    image_pixels = image_pixels./image_pixels(3,:);
    
    for j=1:size(image_pixels,2)
        u = round(image_pixels(1,j));
        v = round(image_pixels(2,j));
        if (u > 0 & u <= n & v > 0 & v <= m)
            plot(u, v, 'o', 'LineWidth', 4, 'MarkerSize', 1, 'Color', [1,0,0]);
        end
    end
    break;
end