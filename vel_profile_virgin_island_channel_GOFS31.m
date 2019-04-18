
%% User input

url_glider291 = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng291-20180701T0000/ng291-20180701T0000.nc3.nc';
url_glider467 = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc';
url_glider487 = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng487-20180701T0000/ng487-20180701T0000.nc3.nc';

% GOFS3.1 outout model location
catalog31_ts = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';
catalog31_uv = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/uv3z';

date_ini = '18-Jul-2018 00:00:00';
date_end = '25-Aug-2018 00:00:00';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

%% GOFS 3.1

%ncdisp(catalog31);
%{
lat31 = ncread(catalog31,'lat');
lon31 = ncread(catalog31,'lon');
depth31 = ncread(catalog31,'depth');
tim31 = ncread(catalog31,'time'); % hours since 2000-01-01 00:00:00

time31 = tim31/24 + datenum(2000,01,01,0,0,0);
%}

load('GOFS31_dimensions.mat')

% Finding subset of data for time period of interest
if isempty(date_ini)
   tti = time(1);
else
   tti = datenum(date_ini);
end

if isempty(date_end)
   tte = time(end);
else
   tte = datenum(date_end);
end

oktime31 = find(time31 >= tti & time31 < tte);

%%
save('GOFS31_dimensions.mat','lat31','lon31','depth31','time31')

%% Load velocity profiles for grid points in the Virgin Island Channel

lon_pos = [3689 3690];
lat_pos = [1724 1725 1726];

u31(length(depth31),length(time31(oktime31))) = nan;
for t=1:length(time31(oktime31))
    disp([length(time31(oktime31)),t])
    u31(:,t) = squeeze(double(ncread(catalog31_uv,'water_u',[lon_pos(2) lat_pos(2) 1 oktime31(t)],[1 1 inf 1])));
end

