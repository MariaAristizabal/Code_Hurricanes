% Glider/Model Data Comparison

clear all;

%% User input

% Glider data location

% ru22
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc';

% RU33 (MAB + SAB)
lon_lim = [-81 -70];
lat_lim = [30 42];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';

% Initial and final date
date_ini = '11-Sep-2018 00:00:00';
date_end = '12-Sep-2018 00:00:00';

%% Glider Extract

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');
CTD_comment = ncreadatt(gdata,'instrument_ctd','comment');

temperature = double(ncread(gdata,'temperature'));
salinity = double(ncread(gdata,'salinity'));
conductivity = double(ncread(gdata,'conductivity'));
pressure = double(ncread(gdata,'pressure'));
time = double(ncread(gdata,'time'));
time = datenum(1970,01,01,0,0,time);
latitude = double(ncread(gdata,'latitude'));
longitude = double(ncread(gdata,'longitude'));

% Finding subset of data for time period of interest
tti = datenum(date_ini);
tte = datenum(date_end);
ok_time_glider = find(time >= tti & time < tte);

% Final lat and lon of deployment for time period of interest 
% This lat and lon is used to find the grid point in the model output
% we can consider changing it to the average lat, lon for the entire
% deployment
Glat = latitude(ok_time_glider(end));
Glon = longitude(ok_time_glider(end));

%% Thermal lag correction (Garau et al. 2011, Thermal lag correction on slocum CTD glider data)

i=1;
ok = isfinite(pressure(:,i)) & isfinite(temperature(:,i));
pres1 = pressure(ok,i);
temp1 = temperature(ok,i);
cond1 = conductivity(ok,i);
salt1 = salinity(ok,i);
time1 = repmat(time(i),1,length(temp1));

i=2;
ok = isfinite(pressure(:,i)) & isfinite(temperature(:,i));
pres2 = pressure(ok,i);
temp2 = temperature(ok,i);
cond2 = conductivity(ok,i);
salt2 = salinity(ok,i);
time2 = repmat(time(2),1,length(temp2));

params = findThermalLagParams(time1, cond1', temp1', pres1', time2, cond2', temp2', pres2');
   
%%
cond1 = 4.19:0.03:4.5;                          
pres1 = 1:11;
temp1 = sort(14:24,'descend');
time1 = datenum(2018,08,01,13,0,0):datenum(0,0,0,0,0,10):datenum(2018,08,01,13,1,40);
salt1 = sw_salt(cond1*10/sw_c3515,temp1,pres1);

cond2 = cond1 + 0.02;
pres2 = pres1;
temp2 = temp1 + 0.1;
time2 = datenum(2018,08,01,13,1,50):datenum(0,0,0,0,0,10):datenum(2018,08,01,13,3,30);
salt2 = sw_salt(cond2*10/sw_c3515,temp2,pres2);

params = findThermalLagParams(time1, cond1, temp1, pres1, ...
                              time2, cond2, temp2, pres2);
%{
if strcmp(CTD_comment,'pumped CTD') 
    params = findThermalLagParams(profile_1_time, profile_1_cond, profile_1_temp, profile_1_press, profile_2_time, profile_2_cond, profile_2_temp, profile_2_press);
    [temp_inside cond_corr] = correctThermalLag(profile_2_time, profile_2_cond, profile_2_temp, params);
    [temp_inside cond_corr] = correctThermalLag(profile_1_time, profile_1_cond, profile_1_temp, params);
    profile_1_sal = sw_salt(cond_corr, temp_inside, profile_1_press/10);
else 
    
    
end
%}    

%% Figure

figure
plot(cond1,-pres1,'.-')
hold on
plot(cond2,-pres2,'.-')

%% 

figure
plot(salt1,temp1,'o-')
hold on
plot(salt2,temp2,'o-')

