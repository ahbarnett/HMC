% 1D demo phase-space anim for HMC, for intuition. MATLAB, self-contained script
% Also tests a variable-scale pdf: log-normal.
% Barnett for Chirag & Bob project. 2/12/21

clear
s = 0.7;                % std dev in log q, difficulty parameter (1.4 = hard)
Vpos = @(q) log(q) + 0.5 * (log(q)/s).^2;   % log-normal centered at log(q)=0
V = @(q) neginf(Vpos(q),q);
gradV = @(q) (1+real(log(q))/s^2)./q;
hessV = @(q) (-1+(1-real(log(q)))/s^2)./q.^2;
x = 1.3; h = 1e-5; fprintf('chk gradV: %.3g\n',gradV(x)-(V(x+h)-V(x-h))/(2*h))
fprintf('chk hessV: %.3g\n',hessV(x)-(gradV(x+h)-gradV(x-h))/(2*h))
hs = hessV(exp(2*[-1 1]*s)); fprintf('hessV @ -+2sig = %.3g, %.3g\n',hs(1),hs(2))
Pi = @(q) exp(-V(q)) / (sqrt(2*pi)*s);      % norm'ed target dens
cdf = @(q) (1+erf(log(q)/(sqrt(2)*s)))/2;    % for norm case
Pilogq = @(l)  exp(-0.5*(l/s).^2) / (sqrt(2*pi)*s);  % ~ N(0,s^2)

H = @(q,p) V(q) + p.^2/2;

figure(1); clf; subplot(2,1,1);
qmax = 3; pmax = 3;    % plot region
qg = linspace(0,qmax,3e3); plot(qg,Pi(qg),'-'); hold on;
plot(qg,V(qg)-min(V(qg)),':');
plot(qg,cdf(qg),'--'); legend('\pi(q)','shifted V(q)','cdf'); %hline(1);
xlabel('q'); title(sprintf('target density: s=%.3g',s)); axis([0 qmax 0 1]);
subplot(2,1,2);
qg = qmax*(0:0.003:1); pg = pmax*(-1:0.02:1);
[q p] = ndgrid(qg,pg);
Hplt = H(q,p)'; contourf(qg,pg,exp(-Hplt),30); colormap(hot(256));
colorbar; title('phase space: \pi(q,p) unnorm'); xlabel('q'); ylabel('p');
axis tight; oaxis = axis; hold on;

eps = 0.05;    % alg params
L = 100;
N = 1000;       % MCMC steps
psi = 1.0 * pi/2;  % mixing ang: pi/2=total randomize p
q=1; p=1;  % initial, somewhere in the mass

nacc = 0; qa = nan(1,N); pa=qa; Ha=qa;   % history arrays
verb = -1;      % -1 = only plot MCMC pts at end; 0=anim pts; 1, 2 = more
for k=1:N
  plot(q,p,'c.','markersize',10);
  if verb>=0 && mod(k,10)==0, axis(oaxis); drawnow; end
  [q1,p1] = leapfrog(q,p,eps,L,gradV,verb);
  qa(k) = q; pa(k)=p; Ha(k)=H(q,p);   % save
  if log(rand()) < H(q,p) - H(q1,p1), q=q1; p=p1; nacc=nacc+1; end    % accept
  pold = p;
  p = cos(psi)*p + sin(psi)*randn();       % mix
  if verb>0, plot([q q],[pold p],'g.-','markersize',10); end   % show p-rand
end
axis(oaxis);
[Ct t t0 ESS] = autoc(Ha);
fprintf('rej ratio = %.3g \t tau0 = %.3g leapsteps, ESS = %.3g\n',1-nacc/N,L*t0,ESS);

figure(2); clf; subplot(2,1,1); plot(Ha,'+-'); title('H vs chain steps');
subplot(2,1,2); plot(t*L,Ct,'-'); xlabel('leap steps');
title(sprintf('autoc of H: tau_0=%.3g leap steps, ESS=%.3g',t0*L,ESS))

figure(3); clf;       % hist & cdf wrt q (could also do log q)
bedge = [0 exp(-2*s:0.3:2*s)]; qmax = bedge(end);   % bins to match log-normal
cou = histc(qa,bedge); cou = cou(1:end-1);   % top entry trash
q = linspace(0,qmax,1e3);
le = log(bedge);
lcen = (le(1:end-1)+le(2:end))/2; lwid = le(2:end)-le(1:end-1);
subplot(1,2,1); bar(lcen,cou./lwid/N); hold on;
errorbar(lcen,cou./lwid/N,sqrt(cou)./lwid/N,'+');   % Poisson errs on counts
%for i=1:numel(bedge)-1, patch(log(bedge([i i+1 i+1 i])),[0 0 1 1]*cou(i)/bwid(i)/N,'red'); hold on; end   % hack for variable width bars
l=log(q); plot(l,Pilogq(l),'-','linewidth',2); set(gca,'xlim',[le(2), le(end)])
title('\pi and empirical pdf wrt l=log q'); xlabel('l=log q')
subplot(1,2,2); ecdf=cumsum(cou)/N;
errorbar(bedge(2:end),ecdf,1.22/sqrt(ESS)*ones(1,numel(bedge)-1),'+-');
hold on; plot(q,cdf(q),'-');
title('cdf and empirical cdf (w/ alpha=0.1 KS lims for ESS)');
set(gca,'xsc','log'); axis([bedge(2) bedge(end) 0 1]); xlabel('q')
% for K_alpha see: https://blogs.sas.com/content/iml/2019/05/20/critical-values-kolmogorov-test.html
% cdf no vert jacobean change under rescaling x-axis



%%%%%% helpers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v = neginf(u,x)
  v=u; v(x<=0) = inf;
end

function [q,p] = leapfrog(q,p,eps,L,gradV,verb)
% q,p can be row vecs of pts. plotting (verb>0) only for 1d. Simple Verlet.
% verb=1: one line per Verlet. verb=2: show p & q shears.
  c=[0.5 0.7 1]; ms=20;  % for plot
  for l=1:L            % not the efficient way but who cares for now
    pold1 = p;   % for plot
    p = p - (eps/2)*gradV(q);
    if abs(p)>5, fprintf('cool it! bad p=%.3g\n',p); p = randn(); end  % hack
    if verb>1, plot([q q],[pold1 p],'-','color',c); end
    qold = q;
    q = q + eps*p;
    if q<=0, fprintf('ouch! bad q=%.3g\n',q); q=1; p=randn(); end   % hack to recover from disaster for log-normal
    if verb>1, plot([qold q],[p p],'-','color',c); end
    pold = p;
    p = p - (eps/2)*gradV(q);
    if verb>1, plot([q q],[pold p],'-','color',c); end
    if verb==1, plot([qold q], [pold1 p], '.-', 'markersize',ms,'color',c); end
  end
end

function [Ctau tau tau0 ESS] = autoc(v)   % v is row vec, scalar for now
v = v(:)';
  v = v-mean(v);
  N = numel(v);
  tau = 0:N/2;
  v = [v, 0*v];   % zero pad
  Ctau = ifft(abs(fft(v)).^2);
  Ctau = Ctau/Ctau(1);
  Ctau = Ctau(1:numel(tau));
  i = find(Ctau<0.1);
  tau0 = i(1)-1;         % 1st dip below 0.1, in index units
  ESS = N/tau0;   % crude (should use integral of C)
end
