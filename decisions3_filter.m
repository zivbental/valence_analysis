function [decisionsLeftPC , decisionsTotal]=decisions3_filter(inputVector,~)

%Adam CC 20050510
%Finds transitions, counts long-interval transitions and reversal as 
%decisions, but using ternary position data.  Counts decisions and calculates their ratio

% % for testing
% clear all
% close all
% clc
% 
% cd D:\matlabData\mats
% load('20060920_1537_2XOP.mat');
% 
% inputVector=fb.data.cX(:,1);
% gaussfwhm=50;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Modified by Alistair Muldal 25/03/11

%Defaults

% Decision zone:
% 
%                    <-0.4->
%  __________________________________________
% [_________________|_______|________________]
% 
% Filter zone:
% 
%              <----2*halfwidth---->
%  __________________________________________
% [___________|____________________|_________]

%Minimum number of times the fly must enter/exit filter window in order
%to be counted
mindecs=1;

halfwidth=0.3;

%If it fails to do so, decision score gets assigned NaN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if size((inputVector), 2)>1
    error('Input data must be a single column vector');
end


%(Gaussian smoothing is now redundant, since data are already smoothed by
%Savitzky-Golay filter in read_file_alistair)

% % if nargin<2
% %     gaussfwhm=50;
% % end

% %smooth position data using gaussian-biased moving average
% %x=spm_conv(inputVector, gaussfwhm);

x=inputVector;

%set up ternary data, and redfine as three unique values
ternaryPositionData=ternarylocationfunc(x);
ternaryPositionData(ternaryPositionData==1) = 3;
ternaryPositionData(ternaryPositionData==-1) = -2;

ternaryFilterData=ternarylocationfunc_filter(x,halfwidth);
ternaryFilterData(ternaryFilterData==1) = 3;
ternaryFilterData(ternaryFilterData==-1) = -2;

%take transitions and then find center entries and exits only
t2 = diff(ternaryPositionData,1,1);
transvec = t2(t2~=0);

t2f = diff(ternaryFilterData,1,1);
transvecf = t2f(t2f~=0);

%create a character array using 'm' as the centerpoint
tr2=char(transvec + double('m'))';

tr2f=char(transvecf + double('m'))';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Count the number of reversal entries and fullway transition entries

%For the decision zone
Left2RightFull=length(strfind(tr2,'op'));
Right2LeftFull=length(strfind(tr2,'jk'));
LeftReversal=length(strfind(tr2,'ok'));
RightReversal=length(strfind(tr2,'jp'));

%For the filter zone:
Left2RightFullf=length(strfind(tr2f,'op'));
Right2LeftFullf=length(strfind(tr2f,'jk'));
LeftReversalf=length(strfind(tr2f,'ok'));
RightReversalf=length(strfind(tr2f,'jp'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Count decisions
LeftDecisions=(Right2LeftFull + LeftReversal);
RightDecisions=(Left2RightFull + RightReversal);
TotalDecisions=(LeftDecisions + RightDecisions);

%For the filter window
LeftDecisionsf=(Right2LeftFullf + LeftReversalf);
RightDecisionsf=(Left2RightFullf + RightReversalf);
TotalDecisionsf=(LeftDecisionsf + RightDecisionsf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%take the percentage and plug this and the total decisions into the
%fbmetrics structure

%Filtering condition
if TotalDecisionsf<mindecs
    leftDecisionsRatioPC=NaN;
    decisionsTotal=NaN;
else
    leftDecisionsRatioPC=((LeftDecisions/TotalDecisions)*100);
end
    decisionsTotal=TotalDecisions;
    decisionsLeftPC=leftDecisionsRatioPC;





% %for plotting
%     if idx==6
%         figure;
%         plot(x);
%         hold on;
%         plot(ternaryPositionData,'g');
%     end;

% %an attempt to use velocity and changes in velocity to find direction
% %reversals
% vel=diff(x);
% vel2=vertcat(0,vel);
% vel2a=abs(vel2);
% vel3=(25*vel2);
% 
% vel4=binarylocationfunc(vel2);
% vel4a=diff(vel4);

% % % acc=diff(vel2);
% % % acc2000=acc * 2000;
% % % 
% % % 
% % % z=find((vel2a) < 0.000075);
% % % zz=zeros(size(x));
% % % zz(z)=1;

% figure;
% plot(x(:,9));
% hold on;
% plot(ternaryPositionData,'g');
% % axis([0 25000 -2.1 3.2]);