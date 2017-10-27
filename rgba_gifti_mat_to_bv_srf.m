% Convert RGBA gifti mat file to BrainVoyager SRF format.
clear all;

% Load extracted gifti mesh mat file
fname_giftimat = '/home/faruk/Git/parcellation/sample_data/derived/rh_rgba.mat';
gmat = load(fname_giftimat);

% Derived parameters
[path, name, ~] = fileparts(fname_giftimat);
verts = gmat.vertices;
faces = gmat.faces;
nr_verts = size(verts, 1);
nr_faces = size(faces, 1);
center = 127.75;
range = max(verts(:)) - min(verts(:));
mid = mean(verts, 1);
verts(:, 1) = verts(:, 1) + center - mid(1);
verts(:, 2) = verts(:, 2) + center - mid(2);
verts(:, 3) = verts(:, 3) + center - mid(3);
vert_rgb = zeros(nr_verts, 4);
vert_rgb(:, 1) = nan(nr_verts, 1);

% Create a surface file
srf = xff('SRF');
srf.ExtendedNeighbors = 2;
srf.NrOfVertices = nr_verts;
srf.NrOfTriangles = nr_faces;
srf.MeshCenter = [center center center];
srf.VertexCoordinate = double(verts);
srf.VertexColor = vert_rgb;
srf.TriangleVertex = double(faces);

% Correct neighbours field is required for shading, otherwise you will see
% random black patches etc.
srf = srf.UpdateNeighbors; 

% Calculate normals
srf = srf.RecalcNormals;

% Fill vertex colors
temp = gmat.rgba;
temp = uint8(temp);
srf.VertexColor(:, 2) = temp(:, 1);  % red
srf.VertexColor(:, 3) = temp(:, 2);  % green
srf.VertexColor(:, 4) = temp(:, 3);  % blue

% Save
fname_out = fullfile(path, [name '.srf']);
srf.SaveAs(fname_out);
disp('Finished')
