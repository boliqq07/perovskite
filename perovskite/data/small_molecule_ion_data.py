# -*- coding: utf-8 -*-

# @Time    : 2021/7/18 16:55
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from ase import Atom, Atoms
from ase.visualize import view


def _generator_molecule(name=None):
    if name is None:
        for i, j in small_molecule.items():
            yield i, Atoms(symbols=j["atom"])
    else:
        for i, j in small_molecule.items():
            if i in name:
                yield i, Atoms(symbols=j["atom"])


def list_molecule(name=None):
    return list(_generator_molecule(name=name))


small_molecule = {

    'NH4': {
        "r_eff": 1.46,
        'atom':
            [
                Atom('H', [-0.7331150000000002, 0.7380750000000003, -0.042284999999999684], index=0),
                Atom('H', [0.7331149999999997, -0.05951500000000021, -0.7509449999999998], index=1),
                Atom('H', [-0.03124500000000019, -0.7380750000000003, 0.7509449999999998], index=2),
                Atom('H', [0.7237249999999991, 0.6993150000000004, 0.6681249999999999], index=3),
                Atom('N', [0.17257499999999926, 0.15980499999999953, 0.1567350000000003], index=4),
            ]
    },

    '(CH3)2NH2': {
        "r_eff": 2.72,
        'atom':
            [
                Atom('H', [1.3282950000000007, 1.0853550000000003, -0.2375949999999989], index=0),
                Atom('H', [1.1685550000000013, 1.4628650000000003, 1.2770650000000003], index=1),
                Atom('C', [1.0659949999999974, 0.788065, 0.623685], index=2),
                Atom('H', [1.5468650000000004, 0.011805000000000287, 0.846425], index=3),
                Atom('H', [-0.7229950000000009, 0.23595500000000014, 1.3067650000000004], index=4),
                Atom('N', [-0.3640149999999984, 0.38695499999999994, 0.5316150000000004], index=5),
                Atom('H', [-0.7902450000000023, 1.0853550000000003, 0.23759500000000067], index=6),
                Atom('H', [-0.3530850000000001, -0.46008499999999986, -1.3067650000000004], index=7),
                Atom('C', [-0.6221150000000009, -0.7290649999999999, -0.4276649999999993], index=8),
                Atom('H', [-0.08406499999999895, -1.4628650000000003, -0.17819499999999877], index=9),
                Atom('H', [-1.5468650000000004, -0.9673750000000001, -0.4009450000000001], index=10),
            ]
    },

    'C3N2H5': {
        "r_eff": 2.58,
        'atom':
            [
                Atom('H', [0.17002000000000006, 0.41295499999999974, 2.1134899999999996], index=0),
                Atom('C', [0.06101000000000045, 0.21720499999999987, 1.0722300000000002], index=1),
                Atom('N', [-0.8753199999999994, -0.5504950000000004, 0.6344099999999999], index=2),
                Atom('H', [-1.5395999999999996, -0.9854050000000005, 1.22241], index=3),
                Atom('C', [-0.7931499999999998, -0.6430650000000004, -0.7011000000000003], index=4),
                Atom('H', [-1.4051099999999996, -1.1675250000000004, -1.3155099999999997], index=5),
                Atom('N', [0.18113000000000046, 0.042324999999999946, -1.1236799999999998], index=6),
                Atom('H', [0.4243700000000006, 0.1682649999999999, -2.1134899999999996], index=7),
                Atom('C', [0.7282300000000008, 0.5910749999999996, -0.03757000000000055], index=8),
                Atom('H', [1.5396, 1.167525, -0.14901000000000053], index=9),
            ]
    },

    '(CH2)3NH2': {
        "r_eff": 2.50,
        'atom':
            [
                Atom('H', [0.44619499999999945, -1.6883199999999996, 0.22132999999999958], index=0),
                Atom('H', [1.9018350000000002, 0.6472000000000007, 0.050499999999999545], index=1),
                Atom('C', [0.8257650000000005, 0.5136900000000004, 0.3619799999999995], index=2),
                Atom('H', [0.7159449999999996, 1.0699800000000002, 1.1794299999999995], index=3),
                Atom('H', [-0.3460949999999996, -1.4809899999999996, 1.78443], index=4),
                Atom('C', [-0.0028450000000002085, -1.0062699999999996, 0.7271299999999994], index=5),
                Atom('N', [-1.1641149999999998, -0.5540599999999998, -0.12319000000000013], index=6),
                Atom('H', [-1.6568749999999999, -1.1755399999999996, -0.7465100000000002], index=7),
                Atom('H', [-1.9018349999999997, -0.3319099999999997, 0.49192999999999953], index=8),
                Atom('C', [-0.13762499999999944, 0.7851600000000003, -0.8097500000000002], index=9),
                Atom('H', [-0.6047349999999994, 1.68832, -0.9612000000000003], index=10),
                Atom('H', [0.24960500000000074, 0.6395100000000005, -1.78443], index=11),
            ]
    },

    "(CH3)3NH": {
        'atom':
            [
                Atom('H', [-1.4946799999999998, 0.12935500000000033, -1.3922699999999995], index=0),
                Atom('C', [-0.9271099999999999, -0.624485, -0.87236], index=1),
                Atom('H', [-1.6077399999999997, -1.1486449999999997, -0.23276000000000074], index=2),
                Atom('H', [-0.2942, -1.3238749999999997, -1.6091099999999998], index=3),
                Atom('N', [-0.020890000000000075, 0.09619500000000025, 0.009619999999999962], index=4),
                Atom('H', [-0.6526399999999999, 0.8129949999999999, 0.6404399999999999], index=5),
                Atom('C', [0.71089, -0.824265, 0.9010099999999994], index=6),
                Atom('H', [-0.018549999999999844, -1.375575, 1.4524000000000008], index=7),
                Atom('H', [1.2034199999999995, -0.2035849999999999, 1.6091099999999994], index=8),
                Atom('H', [1.4588199999999993, -1.502295, 0.25905999999999985], index=9),
                Atom('H', [1.6077399999999997, 1.3063150000000006, -0.16047999999999973], index=10),
                Atom('C', [0.98238, 0.800205, -0.8325900000000002], index=11),
                Atom('H', [0.4359599999999997, 1.5022950000000002, -1.4219500000000003], index=12),
                Atom('H', [1.6031899999999997, 0.03313500000000014, -1.50855], index=13),
            ]

    },

    'NH2NH3': {
        "r_eff": 2.17,
        'atom':
            [
                Atom('H', [1.0942949999999998, -1.0588499999999998, 0.8671099999999998], index=0),
                Atom('H', [-0.11931500000000028, 1.0990599999999997, -0.0017000000000000348], index=1),
                Atom('N', [-0.5213850000000004, 0.22858, -0.008890000000000509], index=2),
                Atom('H', [-1.1570650000000002, 0.24924000000000035, 0.8192399999999993], index=3),
                Atom('H', [-1.110875, 0.2351000000000001, -0.8671099999999998], index=4),
                Atom('N', [0.5209450000000002, -1.0990599999999997, 0.021469999999999434], index=5),
                Atom('H', [1.0942949999999998, -1.0588499999999998, 0.8671099999999998], index=6),
                Atom('H', [1.1570650000000002, -1.0624199999999997, -0.7803400000000003], index=7),
            ]
    },

    '(CH3)4N': {
        "r_eff": 2.92,
        'atom':
            [
                Atom('H', [-0.1277100000000022, 1.22663, -1.6666299999999996], index=0),
                Atom('H', [-1.4441199999999998, 1.3102699999999992, -0.86205], index=1),
                Atom('H', [-0.16699999999999982, 1.9235799999999994, -0.27585000000000015], index=2),
                Atom('C', [-0.5010200000000022, 1.1987500000000004, -0.7930799999999998], index=3),
                Atom('N', [-0.15325000000000166, 0.0, -0.13791999999999938], index=4),
                Atom('C', [1.3065999999999978, 0.0, 0.2528699999999997], index=5),
                Atom('H', [1.8174399999999995, 0.0, -0.5517099999999999], index=6),
                Atom('H', [1.5227199999999979, 0.7775200000000009, 0.7586100000000009], index=7),
                Atom('H', [1.5227199999999979, -0.7775200000000009, 0.7586100000000009], index=8),
                Atom('H', [-0.5992599999999992, 0.7775200000000009, 1.6666299999999996], index=9),
                Atom('C', [-0.8743300000000023, 0.0, 1.19538], index=10),
                Atom('H', [-0.5992599999999992, -0.7775200000000009, 1.6666299999999996], index=11),
                Atom('H', [-0.16699999999999982, -1.9235799999999994, -0.27585000000000015], index=12),
                Atom('C', [-0.5010200000000022, -1.1987500000000004, -0.7930799999999998], index=13),
                Atom('H', [-0.1277100000000022, -1.22663, -1.6666299999999996], index=14),
                Atom('H', [-1.4441199999999998, -1.310270000000001, -0.86205], index=15),
                Atom('H', [-1.8174399999999995, 0.0, 1.1264199999999995], index=16),
            ]
    },

    # 'HC(NH2)2': {
    #     "r_eff": 2.53,
    #     'atom':
    #         []
    # },

    'OHNH3': {
        "r_eff": 2.16,
        'atom':
            [
                Atom('H', [0.7108149999999993, 0.31086999999999954, -1.0767650000000004], index=0),
                Atom('H', [-0.8047250000000004, 1.0268899999999999, -0.2607649999999997], index=1),
                Atom('N', [0.24007499999999915, 0.3854199999999999, -0.15964499999999981], index=2),
                Atom('H', [0.8047249999999995, 0.7695799999999995, 0.5632950000000005], index=3),
                Atom('O', [0.2845549999999992, -1.0268900000000003, 0.23176499999999978], index=4),
                Atom('H', [-0.25339500000000026, -0.8992300000000002, 1.0767650000000004], index=5),
            ]
    },

    'C(NH2)3': {
        "r_eff": 2.78,
        'atom':
            [
                Atom('H', [0.020134999999999792, 1.9005349999999996, -0.33272499999999994], index=0),
                Atom('N', [0.05185499999999976, 1.1419350000000001, -0.7779849999999993], index=1),
                Atom('H', [0.11652500000000021, 1.1499949999999997, -1.6554649999999995], index=2),
                Atom('C', [0.005494999999999806, -4.999999999810711e-06, -0.13373499999999972], index=3),
                Atom('N', [-0.08845499999999973, -4.999999999810711e-06, 1.2248850000000004], index=4),
                Atom('H', [-0.11652500000000021, -0.7622650000000002, 1.6554649999999995], index=5),
                Atom('H', [-0.11652500000000021, 0.7622650000000002, 1.6554649999999995], index=6),
                Atom('N', [0.05185499999999976, -1.1419350000000001, -0.7779849999999993], index=7),
                Atom('H', [0.020134999999999792, -1.900535, -0.33272499999999994], index=8),
                Atom('H', [0.11652500000000021, -1.149995, -1.6554649999999995], index=9),
            ]
    },

    'CH3NH3': {
        "r_eff": 2.17,
        'atom':
            [
                Atom('H', [-0.6301450000000002, 1.22889, 0.98712], index=0),
                Atom('H', [1.092165, 1.0952899999999997, 0.1675400000000007], index=1),
                Atom('C', [0.2018449999999996, 0.54816, 0.5806400000000007], index=2),
                Atom('H', [0.4528349999999999, 0.07213999999999987, 1.38537], index=3),
                Atom('N', [-0.2508250000000003, -0.5750400000000001, -0.5839100000000004], index=4),
                Atom('H', [-0.4971350000000001, -0.17771000000000003, -1.38537], index=5),
                Atom('H', [-1.092165, -1.1285100000000001, -0.2318099999999994], index=6),
                Atom('H', [0.5239150000000001, -1.22889, -1.0215199999999998], index=7),
            ]
    },

    'CH3CH2NH3': {
        "r_eff": 2.74,
        'atom':
            [
                Atom('H', [0.09842999999999957, 1.1531650000000004, 1.8592799999999996], index=0),
                Atom('H', [1.4246500000000002, 0.6870349999999998, 1.03973], index=1),
                Atom('N', [0.4132199999999995, 0.4588749999999999, 1.2393800000000001], index=2),
                Atom('H', [0.3017999999999996, -0.49283500000000036, 1.7707099999999998], index=3),
                Atom('H', [-1.4246500000000004, 0.3514149999999998, 0.24651999999999985], index=4),
                Atom('H', [-0.19161000000000072, 1.5507150000000003, -0.5487099999999998], index=5),
                Atom('C', [-0.36756000000000055, 0.5215749999999995, -0.05875999999999992], index=6),
                Atom('C', [0.10324999999999962, -0.5283550000000004, -0.9276399999999998], index=7),
                Atom('H', [-0.09022000000000041, -1.550715, -0.4188099999999997], index=8),
                Atom('H', [1.1751200000000002, -0.3658050000000004, -1.18723], index=9),
                Atom('H', [-0.4506600000000005, -0.4555150000000001, -1.8592799999999998], index=10),
            ]
    },

    'CH3CH2CH2NH3': {
        # "r_eff": 2.50,
        'atom':
            [
                Atom('H', [-1.2879900000000002, -0.9942500000000003, 0.07674999999999965], index=0),
                Atom('H', [-1.9068, -0.15779000000000032, -2.0735], index=1),
                Atom('C', [-0.8357300000000003, -0.6703500000000002, -1.3637099999999998], index=2),
                Atom('H', [-0.6112200000000003, -1.2600500000000001, -2.2021499999999996], index=3),
                Atom('H', [1.9067999999999996, -0.7029800000000004, -0.5225599999999999], index=4),
                Atom('C', [0.8521399999999995, -0.17620000000000036, -1.2308899999999996], index=5),
                Atom('H', [1.3260899999999998, 0.09742999999999968, -2.6945499999999996], index=6),
                Atom('C', [0.6056900000000001, 0.6868399999999997, -0.15832999999999986], index=7),
                Atom('H', [1.8348899999999997, 0.9941499999999999, 0.023100000000000342], index=8),
                Atom('H', [-0.3588900000000006, 1.2600499999999997, -0.8722799999999995], index=9),
                Atom('H', [0.6721500000000002, -0.2217100000000003, 2.6292999999999997], index=10),
                Atom('H', [-0.02955000000000041, 0.9369199999999998, 2.6945499999999996], index=11),
                Atom('N', [-0.08591000000000015, 0.3776499999999996, 1.90027], index=12),
                Atom('H', [-1.3730700000000002, 0.21079999999999988, 1.90414], index=13),
            ]
    },

    'CH3CH2CH2CH2NH3': {
        # "r_eff": 2.50,
        'atom':
            [
                Atom('H', [-0.5550999999999995, 0.7588499999999998, -0.7690650000000003], index=0),
                Atom('C', [0.31267000000000067, 0.31709999999999994, -0.7585850000000003], index=1),
                Atom('H', [0.42613000000000056, -0.4872599999999998, -1.5706150000000005], index=2),
                Atom('H', [1.9600800000000014, 2.1764600000000005, -0.4038350000000004], index=3),
                Atom('H', [2.905989999999999, 1.00847, -1.5890750000000002], index=4),
                Atom('C', [1.9634, 1.4770900000000005, -1.2929150000000003], index=5),
                Atom('H', [2.2882999999999996, 2.1328700000000005, -2.303685], index=6),
                Atom('C', [-0.3572799999999994, -0.3920899999999998, 0.8460349999999996], index=7),
                Atom('H', [-0.5399299999999991, 0.3582799999999997, 1.7131049999999997], index=8),
                Atom('H', [0.43942000000000014, -0.8990400000000001, 0.8873849999999996], index=9),
                Atom('N', [-1.9863099999999996, -1.5324, 1.3510149999999994], index=10),
                Atom('H', [-2.3128699999999993, -2.17646, 2.3036849999999998], index=11),
                Atom('H', [-2.9059899999999996, -1.1030900000000001, 1.6698949999999995], index=12),
                Atom('H', [-2.005909999999999, -2.14658, 0.48354499999999945], index=13),
            ]
    },

    'CH3C(NH2)2': {
        # "r_eff": 2.50,
        'atom':
            [
                Atom('H', [0.21235000000000026, 0.41484999999999994, -2.014175], index=0),
                Atom('H', [-0.9773599999999996, 1.17603, -1.2926449999999998], index=1),
                Atom('N', [-0.22837999999999958, 0.61808, -1.173835], index=2),
                Atom('C', [-0.018389999999999684, 0.01183999999999985, -0.022935000000000372], index=3),
                Atom('N', [0.8704000000000001, -0.8168100000000003, 0.11776500000000034], index=4),
                Atom('H', [1.4605099999999998, -0.9731200000000002, -0.6235849999999998], index=5),
                Atom('H', [1.1269300000000007, -1.1760300000000004, 1.035475], index=6),
                Atom('C', [-0.7937699999999994, 0.2501599999999997, 1.1380049999999997], index=7),
                Atom('H', [-1.4295299999999997, -0.7185900000000003, 1.4126250000000002], index=8),
                Atom('H', [-1.4605099999999998, 0.9673299999999996, 0.8851849999999999], index=9),
                Atom('H', [-0.08020999999999967, 0.6403599999999998, 2.014175], index=10),
            ]
    },

    '(CH3)2CHNH3': {
        'atom':
            [
                Atom('H', [0.9190249999999995, 1.6731250000000002, -0.18126000000000086], index=0),
                Atom('H', [-0.3315249999999992, 0.587815, -1.4399900000000008], index=1),
                Atom('C', [0.5148650000000004, 0.589035, -0.8910400000000007], index=2),
                Atom('H', [1.380865, 0.34750499999999995, -1.6912500000000006], index=3),
                Atom('C', [0.03620500000000071, -0.583405, -0.07603000000000071], index=4),
                Atom('H', [-0.25588499999999925, -1.673125, -0.8129300000000006], index=5),
                Atom('N', [-1.346005, -0.47319500000000003, 0.8830099999999987], index=6),
                Atom('H', [-1.1893249999999997, 0.4980850000000001, 1.6912500000000001], index=7),
                Atom('H', [-2.134975, -0.502745, 0.32776999999999923], index=8),
                Atom('H', [-1.728205, -1.317805, 1.36651], index=9),
                Atom('C', [0.03620500000000071, -0.583405, -0.07603000000000071], index=10),
                Atom('H', [-0.25588499999999925, -1.673125, -0.8129300000000006], index=11),
                Atom('H', [1.4057849999999998, 0.6479349999999999, 1.60257], index=12),
                Atom('C', [1.1549049999999994, -0.43747499999999984, 0.8472200000000001], index=13),
                Atom('H', [2.1349750000000007, -0.552775, 0.18491999999999909], index=14),
                Atom('H', [0.8114950000000007, -1.2580749999999998, 1.4430999999999994], index=15),
            ]
    },

    'C3H4NS': {
        "r_eff": 3.20,
        'atom':
            [
                Atom('H', [0.0, 2.0021549999999984, -1.0947200000000001], index=0),
                Atom('H', [0.0, 1.932364999999999, -1.1999600000000004], index=1),
                Atom('C', [0.0, 1.173375, -0.6107100000000001], index=2),
                Atom('C', [0.0, 1.173375, 0.6107100000000001], index=3),
                Atom('H', [0.0, 2.0021549999999984, 1.0947199999999997], index=4),
                Atom('H', [0.0, 1.932364999999999, 1.19996], index=5),
                Atom('N', [0.0, -0.1753549999999997, 0.9832399999999994], index=6),
                Atom('S', [0.0, -0.15616500000000144, 1.3834599999999995], index=7),
                Atom('H', [0.0, -0.4318349999999995, 1.8480699999999999], index=8),
                Atom('C', [0.0, -1.044265000000001, 0.0], index=9),
                Atom('H', [0.0, -2.002155, 0.0], index=10),
                Atom('N', [0.0, -0.1753549999999997, -0.9832400000000003], index=11),
                Atom('S', [0.0, -0.15616500000000144, -1.3834600000000004], index=12),
                Atom('H', [0.0, -0.4318349999999995, -1.8480700000000003], index=13),
                Atom('C', [0.0, 1.173375, -0.6107100000000001], index=14),
            ]
    },

    'C4H12N2': {
        "type": "naphthene",
        'atom':
            [
                Atom('H', [0.6430400000000001, 1.8965099999999993, 1.0611799999999998], index=0),
                Atom('N', [0.36270999999999987, 1.3277800000000006, 0.4212799999999999], index=1),
                Atom('C', [-0.9362300000000001, 0.7179800000000007, 0.8323899999999997], index=2),
                Atom('H', [-0.8289200000000001, 0.2693499999999993, 1.6858400000000002], index=3),
                Atom('H', [-1.6025799999999997, 1.41521, 0.93988], index=4),
                Atom('C', [-1.40585, -0.27200000000000024, -0.19901999999999997], index=5),
                Atom('H', [-2.22954, -0.6866299999999992, 0.09950999999999999], index=6),
                Atom('H', [-0.2461599999999997, -1.7971599999999999, 0.3370199999999999], index=7),
                Atom('N', [-0.36270999999999987, -1.3277800000000006, -0.4212799999999999], index=8),
                Atom('H', [-0.6430400000000001, -1.8965099999999993, -1.0611799999999998], index=9),
                Atom('H', [-1.5854999999999997, 0.18765999999999927, -1.0335799999999997], index=10),
                Atom('C', [0.9362300000000001, -0.7179800000000007, -0.8323899999999997], index=11),
                Atom('H', [1.6025799999999997, -1.41521, -0.93988], index=12),
                Atom('H', [0.8289200000000001, -0.2693499999999993, -1.6858400000000002], index=13),
                Atom('H', [2.22954, 0.6866299999999992, -0.09950999999999999], index=14),
                Atom('C', [1.40585, 0.27200000000000024, 0.19901999999999997], index=15),
                Atom('H', [1.5854999999999997, -0.18765999999999927, 1.0335800000000002], index=16),
                Atom('H', [0.2461599999999997, 1.7971599999999999, -0.3370199999999999], index=17),
                Atom('N', [0.36270999999999987, 1.3277800000000006, 0.4212799999999999], index=18),
                Atom('H', [0.6430400000000001, 1.8965099999999993, 1.0611799999999998], index=19),
            ]
    },

    'N(CN)2': {
        'atom':
            [
                Atom('N', [-2.089224999999999, 0.0, 0.8555700000000002], index=0),
                Atom('C', [-1.1857749999999987, 0.0, 0.11965999999999966], index=1),
                Atom('N', [-0.3468549999999988, 0.0, -0.8555700000000002], index=2),
                Atom('C', [0.9115150000000014, 0.0, -0.6940300000000001], index=3),
                Atom('N', [2.0892250000000008, 0.0, -0.7119800000000001], index=4),
            ]
    },

    'N3': {
        'atom':
            [
                Atom('N', [-0.9061950000000003, 0.0, 0.7427550000000007], index=0),
                Atom('N', [-0.025554999999999772, 0.0, 0.0286249999999999], index=1),
                Atom('N', [0.9061950000000003, 0.0, -0.7427550000000003], index=2),
            ]
    },

    'CN': {
        'atom':
            [
                Atom('C', [0.5742849999999997, 0.0, -0.03742499999999982], index=0),
                Atom('N', [-0.5742850000000006, 0.0, 0.03742499999999982], index=1),
            ]
    },

    'HCOO': {
        "r_eff": 1.35,
        "r_h": 447,
        'atom':
            [
                Atom('O', [0.2301899999999999, -0.11942499999999967, -1.0185350000000009], index=0),
                Atom('C', [0.08782999999999985, 0.01346500000000006, 0.2493149999999993], index=1),
                Atom('O', [-0.5489700000000002, -0.6958750000000009, 1.018535], index=2),
                Atom('H', [0.5489700000000002, 0.6958750000000009, 0.6581549999999989], index=3),
            ]
    },

    # "C7H7" 苯甲基
    # "(HN)(CH2)3S" 环
    # "N(C3H7)4" 不要，删去
    # "NC4H8"  环
    # "CH(NH2)2" 不要，删去

    'C10H15NF': {
        'atom':
            [
                Atom('F', [3.9632, 0.0008, -0.5367], index=0),
                Atom('N', [-2.3978, 0.0002, -0.1583], index=1),
                Atom('C', [-1.4367, -0.0002, 1.0217], index=2),
                Atom('C', [-2.1624, -1.2425, -1.0049], index=3),
                Atom('C', [-3.8306, 0.0006, 0.3547], index=4),
                Atom('C', [-2.1615, 1.2426, -1.0048], index=5),
                Atom('C', [0.0048, -0.0008, 0.6058], index=6),
                Atom('C', [0.6665, -1.2085, 0.4148], index=7),
                Atom('C', [0.6656, 1.2074, 0.4154], index=8),
                Atom('C', [2.0067, -1.2079, 0.0282], index=9),
                Atom('C', [2.0058, 1.208, 0.0288], index=10),
                Atom('C', [2.6765, 0.0003, -0.1647], index=11),
                Atom('H', [-1.6646, -0.8821, 1.6348], index=12),
                Atom('H', [-1.6642, 0.8815, 1.6351], index=13),
                Atom('H', [-2.2083, -2.1242, -0.3588], index=14),
                Atom('H', [-2.9591, -1.2936, -1.7544], index=15),
                Atom('H', [-1.2028, -1.1684, -1.5221], index=16),
                Atom('H', [-3.9812, 0.8999, 0.9594], index=17),
                Atom('H', [-4.5092, 0.0009, -0.5035], index=18),
                Atom('H', [-3.9818, -0.8985, 0.9593], index=19),
                Atom('H', [-2.2068, 2.1243, -0.3587], index=20),
                Atom('H', [-1.2021, 1.1678, -1.5221], index=21),
                Atom('H', [-2.9583, 1.2942, -1.7543], index=22),
                Atom('H', [0.162, -2.1577, 0.5718], index=23),
                Atom('H', [0.1607, 2.1563, 0.5729], index=24),
                Atom('H', [2.5297, -2.1479, -0.1206], index=25),
                Atom('H', [2.5282, 2.1485, -0.1195], index=26),
            ]
    },
    'C10H16N': {
        'atom':
            [
                Atom('N', [-2.0458, 0, 0.0963], index=0),
                Atom('C', [-1.0223, -0.0003, -1.03], index=1),
                Atom('C', [-1.856, 1.2428, 0.954], index=2),
                Atom('C', [-3.4488, 0, -0.4934], index=3),
                Atom('C', [-1.8563, -1.2424, 0.9546], index=4),
                Atom('C', [0.3945, -0.0003, -0.5368], index=5),
                Atom('C', [1.0448, 1.2079, -0.3102], index=6),
                Atom('C', [1.045, -1.2082, -0.3101], index=7),
                Atom('C', [2.3621, 1.2081, 0.1486], index=8),
                Atom('C', [2.3622, -1.2079, 0.1488], index=9),
                Atom('C', [3.0207, 0.0003, 0.3781], index=10),
                Atom('H', [-1.2165, 0.8813, -1.6548], index=11),
                Atom('H', [-1.2165, -0.8824, -1.6543], index=12),
                Atom('H', [-1.8665, 2.1243, 0.3062], index=13),
                Atom('H', [-2.6923, 1.2945, 1.6593], index=14),
                Atom('H', [-0.9261, 1.1684, 1.5226], index=15),
                Atom('H', [-3.5668, -0.8993, -1.1052], index=16),
                Atom('H', [-4.173, 0.0002, 0.3267], index=17),
                Atom('H', [-3.5667, 0.8991, -1.1056], index=18),
                Atom('H', [-1.8669, -2.1241, 0.3073], index=19),
                Atom('H', [-0.9261, -1.1678, 1.523], index=20),
                Atom('H', [-2.6924, -1.2934, 1.6599], index=21),
                Atom('H', [0.5493, 2.1569, -0.4947], index=22),
                Atom('H', [0.5497, -2.1573, -0.4945], index=23),
                Atom('H', [2.8757, 2.1487, 0.3246], index=24),
                Atom('H', [2.8761, -2.1483, 0.3252], index=25),
                Atom('H', [4.0468, 0.0004, 0.7343], index=26),
            ]
    },
    'C3H10N(CH3CH2CH2NH3)': {
        'atom':
            [
                Atom('N', [2.5369, 0, 0], index=0),
                Atom('C', [3.403, -0.5, 0], index=1),
                Atom('C', [4.269, 0, 0], index=2),
                Atom('C', [5.135, -0.5, 0], index=3),
                Atom('H', [3.0044, -0.9749, 0], index=4),
                Atom('H', [3.8015, -0.9749, 0], index=5),
                Atom('H', [4.6675, 0.4749, 0], index=6),
                Atom('H', [3.8705, 0.4749, 0], index=7),
                Atom('H', [2, 0.31, 0], index=8),
                Atom('H', [2.2269, -0.5369, 0], index=9),
                Atom('H', [2.8469, 0.5369, 0], index=10),
                Atom('H', [4.825, -1.0369, 0], index=11),
                Atom('H', [5.672, -0.81, 0], index=12),
                Atom('H', [5.445, 0.0369, 0], index=13),
            ]
    },
    'C3H10N(CH3NH(CH3)2': {
        'atom':
            [
                Atom('N', [0, 0.0001, -0.3263], index=0),
                Atom('C', [1.0818, 0.947, 0.1088], index=1),
                Atom('C', [0.2793, -1.4103, 0.1087], index=2),
                Atom('C', [-1.3611, 0.4632, 0.1088], index=3),
                Atom('H', [0.0001, 0.0001, -1.3553], index=4),
                Atom('H', [1.108, 0.9697, 1.2015], index=5),
                Atom('H', [2.0305, 0.5882, -0.2988], index=6),
                Atom('H', [0.8516, 1.935, -0.2985], index=7),
                Atom('H', [0.2859, -1.4443, 1.2015], index=8),
                Atom('H', [-0.5057, -2.0525, -0.2989], index=9),
                Atom('H', [1.2501, -1.7048, -0.2986], index=10),
                Atom('H', [-1.5248, 1.4643, -0.2985], index=11),
                Atom('H', [-2.1014, -0.2302, -0.2988], index=12),
                Atom('H', [-1.3939, 0.4742, 1.2016], index=13),
            ]
    },

    'C4H12N(CH3CH2CH2CH2NH3)': {
        'atom':
            [
                Atom('N', [-2.4645, 0.3523, -0.0146], index=0),
                Atom('C', [-0.0171, 0.3441, 0.0274], index=1),
                Atom('C', [-1.2674, -0.5224, 0.0015], index=2),
                Atom('C', [1.2463, -0.5151, -0.0016], index=3),
                Atom('C', [2.5027, 0.3412, -0.0126], index=4),
                Atom('H', [-0.0164, 0.9712, 0.9275], index=5),
                Atom('H', [-0.018, 1.0221, -0.8353], index=6),
                Atom('H', [-1.3178, -1.1434, -0.8978], index=7),
                Atom('H', [-1.3464, -1.1545, 0.8909], index=8),
                Atom('H', [1.2416, -1.1551, -0.8917], index=9),
                Atom('H', [1.2672, -1.1757, 0.8729], index=10),
                Atom('H', [-2.4779, 0.9622, -0.843], index=11),
                Atom('H', [-2.5045, 0.9561, 0.8173], index=12),
                Atom('H', [-3.3293, -0.2044, -0.0306], index=13),
                Atom('H', [2.5546, 0.9728, 0.8801], index=14),
                Atom('H', [2.5282, 0.9893, -0.8945], index=15),
                Atom('H', [3.394, -0.2937, -0.0318], index=16),
            ]
    },
    'C4H7F2N': {
        'atom':
            [
                Atom('F', [-1.7362, 1.0626, -0.0007], index=0),
                Atom('F', [-1.584, -1.1036, 0.001], index=1),
                Atom('N', [2.4524, 0.2671, 0], index=2),
                Atom('C', [1.2064, -0.4833, 0.0005], index=3),
                Atom('C', [0.259, 0.1089, -1.0639], index=4),
                Atom('C', [0.2588, 0.111, 1.0634], index=5),
                Atom('C', [-0.8563, 0.0373, -0.0002], index=6),
                Atom('H', [1.2951, -1.5736, 0.0016], index=7),
                Atom('H', [0.4726, 1.133, -1.3939], index=8),
                Atom('H', [0.1006, -0.5188, -1.9455], index=9),
                Atom('H', [0.1004, -0.5149, 1.9462], index=10),
                Atom('H', [0.4724, 1.1358, 1.3915], index=11),
                Atom('H', [3.0387, 0.0848, -0.8136], index=12),
                Atom('H', [3.0384, 0.0863, 0.8141], index=13),
            ]
    },
    'C4H9NH3': {
        'atom':
            [
                Atom('N', [-2.0086, 0.0029, -0.1042], index=0),
                Atom('C', [0.4729, -0.0131, -0.3333], index=1),
                Atom('C', [-0.7762, -0.6901, 0.2567], index=2),
                Atom('C', [0.5449, 1.4409, 0.1407], index=3),
                Atom('C', [1.767, -0.7407, 0.0401], index=4),
                Atom('H', [0.3945, -0.0135, -1.4284], index=5),
                Atom('H', [-0.8379, -1.7207, -0.1102], index=6),
                Atom('H', [-0.7051, -0.7456, 1.349], index=7),
                Atom('H', [0.5155, 1.5077, 1.2338], index=8),
                Atom('H', [1.4765, 1.9094, -0.1964], index=9),
                Atom('H', [-0.2734, 2.0441, -0.2648], index=10),
                Atom('H', [1.916, -0.7517, 1.1251], index=11),
                Atom('H', [2.6351, -0.2559, -0.4187], index=12),
                Atom('H', [1.7386, -1.7784, -0.3083], index=13),
                Atom('H', [-2.0254, 0.1783, -1.1079], index=14),
                Atom('H', [-2.0462, 0.9145, 0.348], index=15),
            ]
    },
    'C8H13N2': {
        'atom':
            [
                Atom('N', [-4.2497, -0.031, 0.2791], index=0),
                Atom('N', [3.6542, -0.014, 0.3798], index=1),
                Atom('C', [-1.9485, 0.0177, -0.6207], index=2),
                Atom('C', [-0.4803, 0.0094, -0.3593], index=3),
                Atom('C', [-2.8123, 0.0182, 0.6446], index=4),
                Atom('C', [0.2122, 1.2134, -0.2324], index=5),
                Atom('C', [0.2001, -1.2026, -0.2407], index=6),
                Atom('C', [1.5853, 1.2056, 0.0131], index=7),
                Atom('C', [1.5732, -1.2105, 0.0048], index=8),
                Atom('C', [2.2659, -0.0063, 0.1317], index=9),
                Atom('H', [-2.2083, -0.8512, -1.2408], index=10),
                Atom('H', [-2.2033, 0.8905, -1.2372], index=11),
                Atom('H', [-2.6157, -0.8567, 1.2721], index=12),
                Atom('H', [-2.6645, 0.9261, 1.2378], index=13),
                Atom('H', [-0.307, 2.1642, -0.3205], index=14),
                Atom('H', [-0.3286, -2.1474, -0.3353], index=15),
                Atom('H', [-4.5226, 0.7795, -0.2925], index=16),
                Atom('H', [-4.8455, -0.0299, 1.1176], index=17),
                Atom('H', [-4.4753, -0.8799, -0.2564], index=18),
                Atom('H', [2.1137, 2.1506, 0.1105], index=19),
                Atom('H', [2.092, -2.1613, 0.0958], index=20),
                Atom('H', [4.1483, -0.8923, 0.4652], index=21),
                Atom('H', [4.157, 0.8587, 0.4713], index=22),
            ]
    },
    'CH(CH3)2': {
        'atom':
            [
                Atom('C', [0, 0.4976, 0], index=0),
                Atom('C', [1.2867, -0.2488, 0], index=1),
                Atom('C', [-1.2867, -0.2488, 0], index=2),
                Atom('H', [0.0001, 1.5534, 0.2487], index=3),
                Atom('H', [1.4792, -0.6772, 0.9877], index=4),
                Atom('H', [1.2732, -1.0535, -0.7408], index=5),
                Atom('H', [2.111, 0.4259, -0.2492], index=6),
                Atom('H', [-1.4793, -0.6771, 0.9877], index=7),
                Atom('H', [-2.111, 0.4261, -0.2492], index=8),
                Atom('H', [-1.2732, -1.0535, -0.7408], index=9),
            ]
    },
    'CH4N2': {
        'atom':
            [
                Atom('N', [1.2059, -0.1866, 0], index=0),
                Atom('N', [-1.1772, -0.2006, 0], index=1),
                Atom('C', [-0.0287, 0.3872, 0], index=2),
                Atom('H', [0.0262, 1.488, -0.0002], index=3),
                Atom('H', [2.0521, 0.3729, 0.0005], index=4),
                Atom('H', [1.3172, -1.1946, 0.0007], index=5),
                Atom('H', [-1.9213, 0.5059, -0.0001], index=6),
            ]
    },
    'N(CH3)4': {
        'atom':
            [
                Atom('N', [0, 0, 0], index=0),
                Atom('C', [-0.9383, -1.1454, 0.3522], index=1),
                Atom('C', [0.6742, 0.5109, 1.2651], index=2),
                Atom('C', [-0.795, 1.1278, -0.642], index=3),
                Atom('C', [1.0591, -0.4933, -0.9752], index=4),
                Atom('H', [-0.3508, -1.9457, 0.8119], index=5),
                Atom('H', [-1.6916, -0.7732, 1.0529], index=6),
                Atom('H', [-1.4132, -1.4995, -0.5675], index=7),
                Atom('H', [1.3437, 1.3324, 0.9934], index=8),
                Atom('H', [-0.1007, 0.8608, 1.9535], index=9),
                Atom('H', [1.2401, -0.3116, 1.7125], index=10),
                Atom('H', [-0.1059, 1.9411, -0.8881], index=11),
                Atom('H', [-1.2719, 0.7432, -1.5484], index=12),
                Atom('H', [-1.5503, 1.4694, 0.0721], index=13),
                Atom('H', [1.7234, 0.3416, -1.2169], index=14),
                Atom('H', [1.6198, -1.3023, -0.4977], index=15),
                Atom('H', [0.5575, -0.8562, -1.8772], index=16),
            ]
    },
    'NH3OH': {
        'atom':
            [
                Atom('O', [0.707, 0, 0], index=0),
                Atom('N', [-0.707, 0, 0], index=1),
                Atom('H', [-1.0639, -0.8553, -0.4439], index=2),
                Atom('H', [-1.0639, 0.8149, -0.5143], index=3),
                Atom('H', [-1.0585, 0.0406, 0.9646], index=4),
                Atom('H', [0.8753, -0.0403, -0.9568], index=5),
            ]
    },


}

if __name__ == "__main__":
    at = list(_generator_molecule(name=["C3H10N(CH3CH2CH2NH3)", ]))
    view(at[0][1])
