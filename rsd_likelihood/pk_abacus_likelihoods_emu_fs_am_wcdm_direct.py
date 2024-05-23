# from FM_likelihoods.pk_likelihood_gc_emu_fs_am_wcdm_redefined import PkLikelihood
from FM_likelihoods.pk_likelihood_cubic_direct_wcdm_redefined import PkLikelihood
from FM_likelihoods.pk_likelihood_cubic_direct_w0wacdm_redefined import PkLikelihood as w0wa_lik

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

class abacus_mock_joint_hex(w0wa_lik):
    pass
