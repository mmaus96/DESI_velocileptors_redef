# from FM_likelihoods.pk_likelihood_gc_emu_fs_am_wcdm_Y1theory import PkLikelihood
from FM_likelihoods.pk_likelihood_Cutsky_emu_fs_am_wcdm_redefined import PkLikelihood
from FM_likelihoods.joint_likelihood_gc_emu_am_wcdm_BBspline_redef_multibin_Y1theory import JointLikelihood
from FM_likelihoods.pk_likelihood_Cutsky_emu_fs_am_wcdm_BBmarg import PkLikelihood as BBlik
# from FM_likelihoods.pk_likelihood_Cutsky_direct_wcdm_redefined import dirLikelihood

class abacus_theory_all_hex(PkLikelihood):
    pass

class abacus_theory_all_rebin_hex(BBlik):
    pass

class abacus_theory_noBGS_folps_hex(PkLikelihood):
    pass

class abacus_theory_BGS_hex(PkLikelihood):
    pass

class abacus_theory_LRG1_hex(BBlik):
    pass

class abacus_mock_Y1theory_FS_BAO(JointLikelihood):
    pass

# class abacus_mock_Y1theory_FS_BAO(JointLikelihood):
#     pass

# class abacus_theory_all_hex(dirLikelihood):
#     pass

# class abacus_theory_all_rebin_hex(dirLikelihood):
#     pass