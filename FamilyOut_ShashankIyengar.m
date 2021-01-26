% Decision Engineering - Homework 3
% Shashank S Iyengar - M12934513
% Bayesian Network - Question 4

close all
clear all
clc

N = 5; 
dag = zeros(N,N);
FO = 1; BP = 2; LO = 3; DO = 4; HB = 5;
dag(FO,[LO DO]) = 1;
dag(BP,DO) = 1;
dag(DO,HB)=1;

false = 1; true = 2;
ns = 2*ones(1,N);

bnet = mk_bnet(dag, ns, 'names', {'familyout','BP','LO','DO','HB'}, 'discrete', 1:5);
names = bnet.names;
bnet.CPD{FO} = tabular_CPD(bnet, FO, [0.85 0.15]);
bnet.CPD{LO} = tabular_CPD(bnet, LO, [0.95 0.4 0.05 0.6]);
bnet.CPD{BP} = tabular_CPD(bnet, BP, [0.99 0.01]);
bnet.CPD{DO} = tabular_CPD(bnet, DO, [0.7 0.1 0.03 0.01 0.3 0.90 0.97 0.99]);
bnet.CPD{HB} = tabular_CPD(bnet, HB, [0.99 0.3 0.01 0.7]);

CPD{FO} = reshape([0.85 0.15], 2, 1);
CPD{LO} = reshape([0.95 0.4 0.05 0.6], 2, 2);
CPD{BP} = reshape([0.99 0.01], 2, 1);
CPD{DO} = reshape([0.7 0.1 0.03 0.01 0.3 0.90 0.97 0.99], 2, 2, 2);
CPD{HB} = reshape([0.99 0.3 0.01 0.7], 2, 2);

%% 3(a): Probability of BP if family is out, light is on and dog barks
engine = jtree_inf_engine(bnet);
evidence = cell(1,N);
evidence{FO} = true;
evidence{LO} = true;
evidence{HB} = true;
[engine, ll] = enter_evidence(engine, evidence);

m = marginal_nodes(engine, BP);
p1 = m.T(true) % P(bp|fo,lo,hb)

%% 3(b): Probability that family is out if light is on and dog barks
engine = jtree_inf_engine(bnet);
evidence = cell(1,N);
evidence{LO} = true;
evidence{HB} = true;
[engine, ll] = enter_evidence(engine, evidence);

m = marginal_nodes(engine, FO);
p2 = m.T(true) % P(fo|lo,hb)

