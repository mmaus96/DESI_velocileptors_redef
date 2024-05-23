from FM_likelihoods.pk_likelihood_gc_emu_fs_am_wcdm_redefined import PkLikelihood
from FM_likelihoods.pk_likelihood_gc_emu_fs_am_wcdm_derived import PkLikelihood as lik_derived
from FM_likelihoods.pk_likelihood_gc_emu_fs_am_lcdm_redefined import PkLikelihood as lik_LCDM
from FM_likelihoods.joint_likelihood_gc_emu_am_wcdm_BBspline_redef_multibin import JointLikelihood as lik_recon

class NGCZ3_Pk(PkLikelihood):
    pass

class SGCZ3_Pk(PkLikelihood):
    pass

class NGCZ1_Pk(PkLikelihood):
    pass

class SGCZ1_Pk(PkLikelihood):
    pass

class abacus_mock_LRG(PkLikelihood):
    pass

class abacus_mock_LRG_hex(PkLikelihood):
    pass

class abacus_mock_ELG(PkLikelihood):
    pass

class abacus_mock_ELG_hex(PkLikelihood):
    pass

class abacus_mock_QSO(PkLikelihood):
    pass

class abacus_mock_QSO_hex(PkLikelihood):
    pass

class abacus_mock_joint_hex(PkLikelihood):
    pass

class abacus_mock_LRG(lik_derived):
    pass

class abacus_mock_LRG_hex(lik_derived):
    pass

class abacus_mock_ELG(lik_derived):
    pass

class abacus_mock_ELG_hex(lik_derived):
    pass

class abacus_mock_QSO(lik_derived):
    pass

class abacus_mock_QSO_hex(lik_derived):
    pass

class abacus_mock_joint_hex(lik_derived):
    pass

class abacus_mock_LRG(lik_LCDM):
    pass

class abacus_mock_LRG_hex(lik_LCDM):
    pass

class abacus_mock_LRG_FS_BAO(lik_recon):
    pass