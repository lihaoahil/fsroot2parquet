// Reference userVars: the body of KStarHyperonExtended::calcUserVars (halld_sim
// AMPTOOLS_AMPS/KStarHyperonExtended.cc) run verbatim on an FSRoot tree.  Writes one
// CSV row per entry: entry, theta, phi, n0, n1, n2, pGamma, bigPhi.
//
//   root -l -b -q 'calc_user_vars_ref.C("tree.root","ntFSGlueX_100000000_100001","",
//                  "POL.root","hPol0",2000,"ref.csv")'
//
// prefix "" reads the kinematic-fit branches (as FSRootDataReader does), "MC" the thrown ones.
#include <fstream>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TLorentzVector.h"
#include "TLorentzRotation.h"
#include "TVector3.h"

void calc_user_vars_ref(const char* fname, const char* tname, const char* prefix,
                        const char* polFile, const char* polHist, Long64_t nmax, const char* out)
{
  TFile* f = TFile::Open(fname);
  TTree* t = (TTree*)f->Get(tname);
  TFile* pf = TFile::Open(polFile);
  TH1D* polFrac_vs_E = (TH1D*)pf->Get(polHist);

  // pKin order of the reaction "Beam Lambda K+ Pi0 Proton Pi-": 0 beam, 2 K+, 3 pi0, 4 p, 5 pi-
  const char* labels[6] = {"PB", "P1", "P2", "P3", "P1a", "P1b"};
  double buf[6][4];
  const char* comp[4] = {"En", "Px", "Py", "Pz"};
  for (int i = 0; i < 6; ++i)
    for (int c = 0; c < 4; ++c)
      t->SetBranchAddress(Form("%s%s%s", prefix, comp[c], labels[i]), &buf[i][c]);

  const int kIndices[2] = {2, 3};
  const int lambdaIndices[2] = {4, 5};
  const double polFraction = 0.;

  std::ofstream csv(out);
  csv << "entry,theta,phi,n0,n1,n2,pGamma,bigPhi\n";
  csv.precision(17);
  Long64_t n = (nmax > 0 && nmax < t->GetEntries()) ? nmax : t->GetEntries();
  for (Long64_t e = 0; e < n; ++e) {
    t->GetEntry(e);
    double* pKin[6];
    for (int i = 0; i < 6; ++i) pKin[i] = buf[i];

    // ---- verbatim from KStarHyperonExtended::calcUserVars ----
    TLorentzVector beam(pKin[0][1], pKin[0][2], pKin[0][3], pKin[0][0]);
    TLorentzVector p1(pKin[kIndices[0]][1], pKin[kIndices[0]][2], pKin[kIndices[0]][3], pKin[kIndices[0]][0]);
    TLorentzVector p2(pKin[kIndices[1]][1], pKin[kIndices[1]][2], pKin[kIndices[1]][3], pKin[kIndices[1]][0]);
    TLorentzVector y1(pKin[lambdaIndices[0]][1], pKin[lambdaIndices[0]][2], pKin[lambdaIndices[0]][3], pKin[lambdaIndices[0]][0]);
    TLorentzVector y2(pKin[lambdaIndices[1]][1], pKin[lambdaIndices[1]][2], pKin[lambdaIndices[1]][3], pKin[lambdaIndices[1]][0]);

    TLorentzVector resonance = p1 + p2;
    TLorentzRotation resonanceBoost(-resonance.BoostVector());
    TLorentzVector hyperon = y1 + y2;
    TLorentzRotation hyperonBoost(-hyperon.BoostVector());

    TLorentzVector hyperon_resonance = resonanceBoost * hyperon;
    TLorentzVector p1_resonance      = resonanceBoost * p1;
    TLorentzVector kstar_hyperon = hyperonBoost * resonance;
    TLorentzVector y1_hyperon = hyperonBoost * y1;

    TVector3 y = (beam.Vect().Unit().Cross(-hyperon.Vect().Unit())).Unit();
    TVector3 z = -1. * hyperon_resonance.Vect().Unit();
    TVector3 x = y.Cross(z).Unit();
    TVector3 angles((p1_resonance.Vect()).Dot(x), (p1_resonance.Vect()).Dot(y), (p1_resonance.Vect()).Dot(z));
    double theta = angles.Theta();
    double phi   = angles.Phi();

    TVector3 yH = (beam.Vect().Unit().Cross(-resonance.Vect().Unit())).Unit();
    TVector3 zH = -1. * kstar_hyperon.Vect().Unit();
    TVector3 xH = yH.Cross(zH).Unit();
    double n0 = cos(y1_hyperon.Vect().Angle(xH));
    double n1 = cos(y1_hyperon.Vect().Angle(yH));
    double n2 = cos(y1_hyperon.Vect().Angle(zH));

    double pGamma;
    if (polFraction > 0.) pGamma = polFraction;
    else {
      int bin = polFrac_vs_E->GetXaxis()->FindBin(pKin[0][0]);
      if (bin == 0 || bin > polFrac_vs_E->GetXaxis()->GetNbins()) pGamma = 0.;
      else pGamma = polFrac_vs_E->GetBinContent(bin);
    }
    TVector3 eps(1.0, 0.0, 0.0);
    double bigPhi = atan2(y.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(y)));
    // ---- end verbatim ----

    csv << e << "," << theta << "," << phi << "," << n0 << "," << n1 << "," << n2 << ","
        << pGamma << "," << bigPhi << "\n";
  }
}
