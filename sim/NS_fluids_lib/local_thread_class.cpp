#include <local_thread_class.H>
#include <AMReX_FileSystem.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX.H>
#include <AMReX_BaseFab.H>
#include <AMReX_Box.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_BLFort.H>
#include <AMReX_Utility.H>
#include <AMReX_Random.H>
#include <AMReX_Print.H>
#include <AMReX_Arena.H>
#include <AMReX_BLBackTrace.H>
#include <AMReX_MemPool.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_Machine.H>
#include <AMReX_ParmParse.H>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <cfenv>
#include <iostream>
#include <iomanip>
#include <new>
#include <stack>
#include <limits>
#include <vector>
#include <algorithm>

int thread_class::nthreads;
double thread_class::number_mfiter_loops=0.0;
std::vector<double> thread_class::tile_d_numPts;
double thread_class::boxarray_d_numPts=0.0;
int thread_class::check_numPts=2;   // ns.loop_check=reserved
int thread_class::fillpatch_barriers=0;
int thread_class::check_selftest=0;
double thread_class::selftest_counter=0.0;
double thread_class::batch_count=0.0;
double thread_class::batch_wcount7=0.0;
double thread_class::batch_wcount11=0.0;
double thread_class::batch_expect=0.0;
double thread_class::batch_wexpect7=0.0;
double thread_class::batch_wexpect11=0.0;
double thread_class::batch_nloops=0.0;

void 
thread_class::Initialize() {

 nthreads=1;

#ifdef _OPENMP
 int tid=0;
 int nthreads_local;
#endif

#ifdef _OPENMP
#pragma omp parallel private(nthreads_local, tid)
{

 /* Obtain thread number */
 tid = omp_get_thread_num();

 /* Only master thread does this */
 if (tid == 0) {
  nthreads_local = omp_get_num_threads();
  nthreads=nthreads_local;
 }
}  /* All threads join master thread and disband */
#endif

 if (nthreads<1)
  amrex::Error("nthreads invalid");

 tile_d_numPts.resize(nthreads);
 for (int tid_local=0;tid_local<nthreads;tid_local++)
  tile_d_numPts[tid_local]=0.0;
 number_mfiter_loops=0.0;

  // main.cpp calls this right after amrex::Initialize, so ParmParse is ready.
 amrex::ParmParse ppns("ns");
 if (ppns.contains("check_loop_counts")||
     ppns.contains("fillpatch_barriers")||
     ppns.contains("check_loop_counts_selftest"))
  amrex::Error("ns.check_loop_counts / ns.fillpatch_barriers were replaced by ns.loop_check = reserved | full | off (and ns.loop_check_selftest)");
 std::string loop_check="reserved";
 ppns.queryAdd("loop_check",loop_check);
 if (loop_check=="full") {
  check_numPts=1; fillpatch_barriers=1;
 } else if (loop_check=="reserved") {
  check_numPts=2; fillpatch_barriers=0;
 } else if (loop_check=="off") {
  check_numPts=0; fillpatch_barriers=0;
 } else
  amrex::Error("ns.loop_check must be reserved, full or off");
 ppns.queryAdd("loop_check_selftest",check_selftest);
 if (check_selftest<0)
  amrex::Error("ns.loop_check_selftest must be >= 0");
 if (amrex::ParallelDescriptor::IOProcessor())
  std::cout << "thread_class: loop_check=" << loop_check <<
    " (check_numPts=" << check_numPts <<
    " fillpatch_barriers=" << fillpatch_barriers <<
    ") loop_check_selftest=" << check_selftest << '\n';

} // end subroutine thread_class::Initialize

void
thread_class::finish_loop(int caller_loop_id,
   const std::string& caller_string) {

 sync_tile_d_numPts();

 if (check_selftest>0) {
  selftest_counter+=1.0;
  if ((selftest_counter==(double) check_selftest)&&
      (amrex::ParallelDescriptor::MyProc()==0)) {
   tile_d_numPts[0]+=1.0;  // pretend this loop visited one cell twice
   std::cout << "thread_class: SELFTEST added 1 fake cell at loop " <<
     selftest_counter << " (" << caller_string << ")\n";
  }
 }

 if (check_numPts==1) {
  amrex::ParallelDescriptor::ReduceRealSum(tile_d_numPts[0]);
  reconcile_d_numPts(caller_loop_id,caller_string); // also counts the loop
 } else if (check_numPts==2) {
   // weights 1..7 and 1..11 by loop position, so a skip in one loop and
   // a double count in another cannot cancel out unnoticed.  Every rank
   // calls finish_loop the same number of times in the same order (the
   // original per-loop Allreduce relied on that too), so the weights and
   // the expected totals agree on all ranks.  All values are integers
   // well below 2^53, so the sums are exact in any order.
  double w7=1.0+std::fmod(batch_nloops,7.0);
  double w11=1.0+std::fmod(batch_nloops,11.0);
  batch_count+=tile_d_numPts[0];
  batch_wcount7+=w7*tile_d_numPts[0];
  batch_wcount11+=w11*tile_d_numPts[0];
  batch_expect+=boxarray_d_numPts;
  batch_wexpect7+=w7*boxarray_d_numPts;
  batch_wexpect11+=w11*boxarray_d_numPts;
  batch_nloops+=1.0;
  number_mfiter_loops=number_mfiter_loops+1.0;
 } else if (check_numPts==0) {
  number_mfiter_loops=number_mfiter_loops+1.0;
 } else
  amrex::Error("check_numPts invalid");

} // end subroutine thread_class::finish_loop

 // check_numPts==2: one Allreduce for all loops since the last flush.
 // Called by every rank at the same point (AmrCore::coarseTimeStep).
void
thread_class::flush_loop_check() {

 if (check_numPts!=2)
  return;

 double sums[3]={batch_count,batch_wcount7,batch_wcount11};
 amrex::ParallelDescriptor::ReduceRealSum(sums,3);
 if ((sums[0]!=batch_expect)||
     (sums[1]!=batch_wexpect7)||
     (sums[2]!=batch_wexpect11)) {
  if (amrex::ParallelDescriptor::IOProcessor()) {
   std::cout << "flush_loop_check: over the last " << batch_nloops <<
     " MFIter loops the tile point total was " << sums[0] <<
     " but the BoxArray total was " << batch_expect <<
     " (weighted: " << sums[1] << " vs " << batch_wexpect7 << ", " <<
     sums[2] << " vs " << batch_wexpect11 << ").\n" <<
     "Rerun with ns.loop_check=full to find the loop.\n";
   std::fflush(NULL);
  }
  amrex::Error("flush_loop_check found tile sum <> boxarray sum\n");
 }
 batch_count=0.0; batch_wcount7=0.0; batch_wcount11=0.0;
 batch_expect=0.0; batch_wexpect7=0.0; batch_wexpect11=0.0;
 batch_nloops=0.0;

} // end subroutine thread_class::flush_loop_check

void 
thread_class::reconcile_d_numPts(int caller_loop_id,
   const std::string& caller_string) {

 number_mfiter_loops=number_mfiter_loops+1.0;

 if (tile_d_numPts[0]==boxarray_d_numPts) {
  // do nothing
 } else {
  amrex::ParallelDescriptor::Barrier();
  std::fflush(NULL);
  amrex::ParallelDescriptor::Barrier();
  for (int pid=0;pid<amrex::ParallelDescriptor::NProcs();pid++) {
   amrex::ParallelDescriptor::Barrier();
   if (amrex::ParallelDescriptor::MyProc()==pid) {
    std::fflush(NULL);
    std::cout << "on processor: " << pid << '\n';
    std::cout << "tile_d_numPts[0]= " << tile_d_numPts[0] << '\n';
    std::cout << "boxarray_d_numPts= " << boxarray_d_numPts << '\n';
    std::cout << "number_mfiter_loops= " << number_mfiter_loops << '\n';
    std::cout << "caller_loop_id= " << caller_loop_id << '\n';
    std::cout << "caller_string= " << caller_string << '\n';
    std::fflush(NULL);
   }
  }  // pid=0..NProcs-1
  amrex::ParallelDescriptor::Barrier();
  std::fflush(NULL);
  amrex::ParallelDescriptor::Barrier();
  amrex::Error("reconcile_d_numPts found tile sum <> boxarray sum\n");
 }

} // end subroutine thread_class::reconcile_d_numPts()

void
thread_class::init_d_numPts(double BA_d_numPts) {

 for (int tid_local=0;tid_local<nthreads;tid_local++) {
  tile_d_numPts[tid_local] = 0.0;
 }
 boxarray_d_numPts=BA_d_numPts;

} // end subroutine thread_class::init_d_numPts

void 
thread_class::sync_tile_d_numPts() {

 for (int tid_local=1;tid_local<nthreads;tid_local++) {
  tile_d_numPts[0]+=tile_d_numPts[tid_local];
 }

} // end subroutine thread_class::sync_tile_d_numPts()
