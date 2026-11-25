
#ifndef LOCAL_THREAD_CLASS_H
#define LOCAL_THREAD_CLASS_H

#include <iostream>
#include <functional>
#include <string>
#include <cstdio>
#include <vector>
#include <memory>


// SUSSMAN
namespace amrex { class MultiFab; }

class thread_class {

 public:
  static void Initialize();
  static void init_d_numPts(double BA_d_numPts);
  static void sync_tile_d_numPts();
  static int nthreads;
  static double number_mfiter_loops;
  static std::vector<double> tile_d_numPts;
  static double boxarray_d_numPts;
  static void reconcile_d_numPts(int caller_loop_id,
    const std::string& caller_string);

   // ns.loop_check = reserved | full | off   (default reserved)
   // How the per-MFIter-loop sanity check (tile cell count == BoxArray
   // cell count, summed over all ranks) is done:
   //  full     = the original: one Allreduce after every loop, abort at
   //             the bad loop; also keeps the Barrier() pairs around
   //             ParallelCopy in amrlib/FillPatchUtil.cpp.  For debugging.
   //  reserved = the same check, held until the end of each coarse time
   //             step: each rank adds up its counts (plus two weighted
   //             copies) over the step's loops, and ONE Allreduce per step
   //             (flush_loop_check, called from AmrCore::coarseTimeStep
   //             before any plotfile/checkpoint of that step is written)
   //             compares them with the BoxArray totals.  A loop that
   //             skips or double-counts cells still aborts the run, at the
   //             end of that step.  No FillPatch barriers (ParallelCopy
   //             already waits for its messages: ParallelCopy_finish).
   //  off      = no check, no barriers; the loop is only counted.
   // Results are bit-identical in all three modes.
   // check_numPts holds the mode: 1 = full, 2 = reserved, 0 = off.
  static int check_numPts;
   // ns.loop_check_selftest=N (default 0): at the N-th loop, add one fake
   // cell to rank 0's count, to prove the check catches it.
  static int check_selftest;
  static double selftest_counter;
   // batch sums for check_numPts==2 (all integer-valued, exact in double)
  static double batch_count;
  static double batch_wcount7;
  static double batch_wcount11;
  static double batch_expect;
  static double batch_wexpect7;
  static double batch_wexpect11;
  static double batch_nloops;
  static void flush_loop_check();

   // Trent Feldman, November 2026: three more switches in the style of
   // ns.loop_check.  full = original behaviour.
   //
   // ns.inner_totals = reserved | full | off   (default reserved)
   //   volWgtSumALL(fast_mode=1) integrates every diagnostic several times
   //   per step; inside a step the solver reads only the maximum temperature
   //   of rigid solids.
   //   reserved = every total is still computed, once per coarse step; which
   //     call computes them rotates with the step number, so every call site
   //     is covered in turn.  The reductions are batched, and the latest
   //     totals stay in NS_sumdata.  With a solid or user-defined sums every
   //     call computes (nothing may be read stale), only batched.
   //   off = not computed unless something reads them.
   //   (combine_state_variable and the mask/recon resizes always run.)
   // ns.mf_check = reserved | full | off   (default reserved)
   //   debug_ngrow, debug_ixType and debug_boxArray: FabArray::ok().
   //   reserved = the same check on every rank at every call (a failing rank
   //     aborts at once), but without the all-reduce; flush_mf_check does one
   //     reduction per coarse step to confirm that all ranks made the same
   //     checks in the same order.
   //   off = no check.
   // ns.facefrac = reserved | full | check   (default reserved)
   //   ColorSum: rebuild the face and cell fractions (FACEFRAC_MM,
   //   FACEFRAC_SOLVE_MM, CELLFRAC_MM).
   //   reserved = rebuild only if an input changed (byte compare of
   //     SLOPE_RECON, MASK_NBR, grids, dmap, time, ...); every
   //     ns.facefrac_audit-th coarse step (default 20, 0=never) each reuse is
   //     also rebuilt and compared byte for byte.
   //   check = audit every reuse.
   // Self-tests (default 0): ns.mf_check_selftest=N (N>0: rank 0 counts check
   //   N twice, caught by flush_mf_check; N<0: rank 1 fails check -N),
   //   ns.facefrac_selftest=1 (change one cached value before each audit).
   //
   // values: 0=off 1=full 2=reserved 3=check
  static int inner_totals,mf_check,facefrac,facefrac_audit;
  static int mf_check_selftest,facefrac_selftest;
  static double mfc_count,mfc_hash;
  static void init_reserved_modes();
  static bool mf_ok(const amrex::MultiFab* mf,int id);
  static void flush_mf_check();
   // 1 = keep the Barrier() pairs around ParallelCopy (loop_check=full)
  static int fillpatch_barriers;
   // Replaces the triple: sync_tile_d_numPts(); ReduceRealSum(...);
   // reconcile_d_numPts(...);  A per-loop MPI call only happens if
   // check_numPts==1.
  static void finish_loop(int caller_loop_id,
    const std::string& caller_string);
};

#endif /*LOCAL_THREAD_CLASS_H*/
