%% Glider/Model Data Comparison

clear all;
 %% User input
 
% Glider data location

% RU22
lon_lim = [120 134];
lat_lim = [30 40];
url_glider = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc';

model_name ='GOFS 3.1';

% GOFS3.1 outout model location
url_model = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

var = 'salinity';

% Initial and final date
date_ini = ' ';  %if empty, date_ini is the firts time stamp in data
date_end = ' ';   %if empty, date_ini is the firts time stamp in data

fig ='yes';
%%
% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'; 

[timeg_vec,depthg_vec,saltg_matrix,timem,depthm,saltm] = ...
          glider_transect_model_comp(url_glider,model_name,url_model,var,fig,date_ini,date_end)
      
%%
var = 'temperature';      
[~,~,tempg_matrix,~,~,tempm] = ...
          glider_transect_model_comp(url_glider,model_name,url_model,var,fig,date_ini,date_end)      


%% Salinity depth=10 
dep = 10;
pos31 = interp1(depthm,1:length(depthm),dep);
posglider = interp1(depthg_vec,1:length(depthg_vec),dep);

figure
set(gcf,'position',[680         630        1059         348])
yvec = 31:0.1:34.5;
xvec = repmat(datenum(2018,08,23,0,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
y = [31 31 34.5 34.5];
x = [datenum(2018,08,17,09,0,0) datenum(2018,08,17,12,0,0) datenum(2018,08,17,12,0,0) datenum(2018,08,17,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
hold on
x = [datenum(2018,08,18,09,0,0) datenum(2018,08,18,12,0,0) datenum(2018,08,18,12,0,0) datenum(2018,08,18,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,19,09,0,0) datenum(2018,08,19,12,0,0) datenum(2018,08,19,12,0,0) datenum(2018,08,19,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,20,09,0,0) datenum(2018,08,20,12,0,0) datenum(2018,08,20,12,0,0) datenum(2018,08,20,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,21,09,0,0) datenum(2018,08,21,12,0,0) datenum(2018,08,21,12,0,0) datenum(2018,08,21,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,22,09,0,0) datenum(2018,08,22,12,0,0) datenum(2018,08,22,12,0,0) datenum(2018,08,22,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,23,09,0,0) datenum(2018,08,23,12,0,0) datenum(2018,08,23,12,0,0) datenum(2018,08,23,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,24,09,0,0) datenum(2018,08,24,12,0,0) datenum(2018,08,24,12,0,0) datenum(2018,08,24,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
h1 = plot(timeg_vec,saltg_matrix(posglider,:),'^-','linewidth',2,'markersize',4);
h2 = plot(timem,saltm(pos31,:),'^-','linewidth',2,'markersize',4);
datetick
set(gca,'fontsize',18)
legend([h1 h2], 'RU22','GOFS 3.1')
title(['Time Series at ',num2str(dep),' m',' Depth'],'fontsize',20)
ylabel('Salinity')
%title('Depth ',num2str(dep),'fontsize',20)

Fig_name = [folder,'Salt_time_series_GOFS31_RU22_',num2str(dep),' m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% Salinity depth=50
dep = 50;
pos31 = interp1(depthm,1:length(depthm),dep);
posglider = interp1(depthg_vec,1:length(depthg_vec),dep);

figure
set(gcf,'position',[680         630        1059         348])
yvec = 32.5:0.1:34.5;
xvec = repmat(datenum(2018,08,23,0,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
y = [32.5 32.5 34.5 34.5];
x = [datenum(2018,08,17,09,0,0) datenum(2018,08,17,12,0,0) datenum(2018,08,17,12,0,0) datenum(2018,08,17,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
hold on
x = [datenum(2018,08,18,09,0,0) datenum(2018,08,18,12,0,0) datenum(2018,08,18,12,0,0) datenum(2018,08,18,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,19,09,0,0) datenum(2018,08,19,12,0,0) datenum(2018,08,19,12,0,0) datenum(2018,08,19,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,20,09,0,0) datenum(2018,08,20,12,0,0) datenum(2018,08,20,12,0,0) datenum(2018,08,20,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,21,09,0,0) datenum(2018,08,21,12,0,0) datenum(2018,08,21,12,0,0) datenum(2018,08,21,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,22,09,0,0) datenum(2018,08,22,12,0,0) datenum(2018,08,22,12,0,0) datenum(2018,08,22,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,23,09,0,0) datenum(2018,08,23,12,0,0) datenum(2018,08,23,12,0,0) datenum(2018,08,23,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,24,09,0,0) datenum(2018,08,24,12,0,0) datenum(2018,08,24,12,0,0) datenum(2018,08,24,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
h1 = plot(timeg_vec,saltg_matrix(posglider,:),'^-','linewidth',2,'markersize',4);
h2 = plot(timem,saltm(pos31,:),'^-','linewidth',2,'markersize',4);
datetick
set(gca,'fontsize',18)
legend([h1 h2], 'RU22','GOFS 3.1')
title(['Time Series at ',num2str(dep),' m',' Depth'],'fontsize',20)
ylabel('Salinity')

Fig_name = [folder,'Salt_time_series_GOFS31_RU22_',num2str(dep),' m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% Temperature depth=10 
dep = 10;
pos31 = interp1(depthm,1:length(depthm),dep);
posglider = interp1(depthg_vec,1:length(depthg_vec),dep);

figure
set(gcf,'position',[680         630        1059         348])
yvec = 15:0.1:30;
xvec = repmat(datenum(2018,08,23,0,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
y = [15 15 30 30];
x = [datenum(2018,08,17,09,0,0) datenum(2018,08,17,12,0,0) datenum(2018,08,17,12,0,0) datenum(2018,08,17,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
hold on
x = [datenum(2018,08,18,09,0,0) datenum(2018,08,18,12,0,0) datenum(2018,08,18,12,0,0) datenum(2018,08,18,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,19,09,0,0) datenum(2018,08,19,12,0,0) datenum(2018,08,19,12,0,0) datenum(2018,08,19,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,20,09,0,0) datenum(2018,08,20,12,0,0) datenum(2018,08,20,12,0,0) datenum(2018,08,20,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,21,09,0,0) datenum(2018,08,21,12,0,0) datenum(2018,08,21,12,0,0) datenum(2018,08,21,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,22,09,0,0) datenum(2018,08,22,12,0,0) datenum(2018,08,22,12,0,0) datenum(2018,08,22,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,23,09,0,0) datenum(2018,08,23,12,0,0) datenum(2018,08,23,12,0,0) datenum(2018,08,23,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,24,09,0,0) datenum(2018,08,24,12,0,0) datenum(2018,08,24,12,0,0) datenum(2018,08,24,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
h1 = plot(timeg_vec,tempg_matrix(posglider,:),'^-','linewidth',2,'markersize',4);
h2 = plot(timem,tempm(pos31,:),'^-','linewidth',2,'markersize',4);
datetick
set(gca,'fontsize',18)
legend([h1 h2], 'RU22','GOFS 3.1')
title(['Time Series at ',num2str(dep),' m',' Depth'],'fontsize',20)
ylabel('Temperature')

Fig_name = [folder,'temp_time_series_GOFS31_RU22_',num2str(dep),' m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% temperature depth=50
dep = 50;
pos31 = interp1(depthm,1:length(depthm),dep);
posglider = interp1(depthg_vec,1:length(depthg_vec),dep);

figure
set(gcf,'position',[680         630        1059         348])
yvec = 10:0.1:20;
xvec = repmat(datenum(2018,08,23,0,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
y = [10 10 20 20];
x = [datenum(2018,08,17,09,0,0) datenum(2018,08,17,12,0,0) datenum(2018,08,17,12,0,0) datenum(2018,08,17,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
hold on
x = [datenum(2018,08,18,09,0,0) datenum(2018,08,18,12,0,0) datenum(2018,08,18,12,0,0) datenum(2018,08,18,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,19,09,0,0) datenum(2018,08,19,12,0,0) datenum(2018,08,19,12,0,0) datenum(2018,08,19,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,20,09,0,0) datenum(2018,08,20,12,0,0) datenum(2018,08,20,12,0,0) datenum(2018,08,20,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,21,09,0,0) datenum(2018,08,21,12,0,0) datenum(2018,08,21,12,0,0) datenum(2018,08,21,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,22,09,0,0) datenum(2018,08,22,12,0,0) datenum(2018,08,22,12,0,0) datenum(2018,08,22,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,23,09,0,0) datenum(2018,08,23,12,0,0) datenum(2018,08,23,12,0,0) datenum(2018,08,23,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,08,24,09,0,0) datenum(2018,08,24,12,0,0) datenum(2018,08,24,12,0,0) datenum(2018,08,24,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
h1 = plot(timeg_vec,tempg_matrix(posglider,:),'^-','linewidth',2,'markersize',4);
h2 = plot(timem,tempm(pos31,:),'^-','linewidth',2,'markersize',4);
datetick
set(gca,'fontsize',18)
legend([h1 h2], 'RU22','GOFS 3.1')
title(['Time Series at ',num2str(dep),' m',' Depth'],'fontsize',20)
ylabel('Salinity')
%title('Depth ',num2str(dep),'fontsize',20)
ylim([10 20])

Fig_name = [folder,'temp_time_series_GOFS31_RU22_',num2str(dep),' m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

