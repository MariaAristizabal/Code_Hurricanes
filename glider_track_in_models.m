%% Along glider track

clear all;

%% User input

% Glider data location

% RAMSES (MAB + SAB)
lon_lim = [-81 -70];
lat_lim = [30 42];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc';

% Initial and final date
%date_ini = '11-Sep-2018 00:00:00';
%date_end = '12-Sep-2018 00:00:00';

% GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Temperature/';

%% Glider Extract

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

time = double(ncread(gdata,'time'));
time = datenum(1970,01,01,0,0,time);
latitude = double(ncread(gdata,'latitude'));
longitude = double(ncread(gdata,'longitude'));

%% GOFS 3.1

%ncdisp(catalog31);

lat31 = ncread(catalog31,'lat');
lon31 = ncread(catalog31,'lon');
depth31 = ncread(catalog31,'depth');
tim31 = ncread(catalog31,'time'); % hours since 2000-01-01 00:00:00

time31 = tim31/24 + datenum(2000,01,01,0,0,0);

oktime31 = find(time31 >= time(1) & time31 < time(end));

% Conversion from glider longitude and latitude to GOFS3.1 convention
target_lon(1:length(longitude)) = nan;
for i=1:length(time)
    if longitude(i) < 0 
       target_lon(i) = 360 + longitude(i);
    else
       target_lon(i) = longitude(i);
    end
end
target_lat = latitude;

sublon31=interp1(time,target_lon,time31(oktime31));
sublat31=interp1(time,target_lat,time31(oktime31));

oklon31=round(interp1(lon31,1:length(lon31),sublon31));
oklat31=round(interp1(lat31,1:length(lat31),sublat31));

target_temp31(length(depth31),length(oktime31))=nan;
for i=1:length(oklon31)
    target_temp31(:,i) = squeeze(double(ncread(catalog31,'water_temp',[oklon31(i) oklat31(i) 1 ok31(i)],[1 1 inf 1])));
end
    
    
%target_salt31 = squeeze(double(ncread(catalog31,'salinity',[oklon31 oklat31 1 ok31(1)],[1 1 inf length(ok31)])));


%% Figure

figure
pcolor(time31(oktime31),-depth31,target_temp31)
shading interp
colormap('jet')
colorbar
ylim([-50 0])
datetick


%% Figure

figure
plot(time,target_lon,'.-k')
hold on
plot(time31(oktime31),sublon31,'*-r')

%% Figure

figure
plot(oklat31,'.-')

%% 
figure
plot(oktime31)
