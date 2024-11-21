import numpy as np

from p2x2p.strategies import utils


ATOL_POWER = 1e-4
RTOL_POWER = 1e-4
ATOL_X_LEVEL = 1e-4
RTOL_X_LEVEL = 1e-4


def run_tests(test_cases):

    # Initialize a list to record test results
    failed_tests = []

    # Run each test
    for (actual, expected, rtol, atol, text) in test_cases:
        try:
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        except AssertionError as e:
            # Record the test number and the error message if the test fails
            failed_tests.append((text, str(e)))

    return failed_tests


def validate_strategy(strategy, params):

    actual_p2x_power = ['running_p2x']
    actual_x2p_power = ['running_x2p', 'running_y2p']
    p2x_up_bids = ['bid_mfrr_up_p2x']
    p2x_down_bids = ['bid_spot_p2x', 'bid_mfrr_down_p2x']
    x2p_up_bids = ['bid_spot_x2p', 'bid_spot_y2p', 'bid_mfrr_up_x2p', 'bid_mfrr_up_y2p']
    x2p_down_bids = ['bid_mfrr_down_x2p', 'bid_mfrr_down_y2p']

    test_cases = []
    storage_init = utils.get_initial_storage_level(params)

    # --- Power values tests ---
    # Ensure that all P2X power values are within the limits
    for key in p2x_up_bids+p2x_down_bids+actual_p2x_power:
        min_power = min([strategy[key].min(), 0])
        max_power = max([strategy[key].max(), params['power_p2x']])
        test_cases += [
            (min_power, 0, RTOL_POWER, ATOL_POWER, f"{key} must be non-negative"),
            (max_power, params['power_p2x'], RTOL_POWER, ATOL_POWER, f"{key} must be less than the P2X power")
        ]
    # Ensure that all X2P power values are within the limits
    for key in x2p_up_bids+x2p_down_bids+actual_x2p_power:
        min_power = min([strategy[key].min(), 0])
        max_power = max([strategy[key].max(), params['power_x2p']])
        test_cases += [
            (min_power, 0, RTOL_POWER, ATOL_POWER, f"{key} must be non-negative"),
            (max_power, params['power_x2p'], RTOL_POWER, ATOL_POWER, f"{key} must be less than the X2P power")
        ]

    # --- Total Power tests ---
    # Ensure that we don't surpass the maximal P2X power
    tot_p2x_power = np.zeros(strategy['bid_spot_p2x'].shape)
    for key in p2x_down_bids:
        tot_p2x_power += strategy[key]
    tot_p2x_power_max = max([tot_p2x_power.max(), params['power_p2x']])
    test_cases += [(tot_p2x_power_max, params['power_p2x'], RTOL_POWER, ATOL_POWER, "The total P2X power must be less than the maximal P2X power")]
    # Ensure that we don't surpass the maximal X2P power
    tot_x2p_power = np.zeros(strategy['bid_spot_x2p'].shape)
    for key in x2p_up_bids:
        tot_x2p_power += strategy[key]
    tot_x2p_power_max = max([tot_x2p_power.max(), params['power_x2p']])
    test_cases += [(tot_x2p_power_max, params['power_x2p'], RTOL_POWER, ATOL_POWER, "The total X2P power must be less than the maximal X2P power")]

    # --- mFRR market bid tests ---
    # Ensure that we don't try to remove P2X power that is not available
    min_p2x_power = min([(strategy['bid_spot_p2x']-strategy['bid_mfrr_up_p2x']).min(), 0])
    test_cases += [(min_p2x_power, 0, RTOL_POWER, ATOL_POWER, "Only P2X power bid to the spot market can be removed")]
    # Ensure that we don't try to remove X2P or Y2P power that is not available
    min_x2p_power = min([(strategy['bid_spot_x2p']-strategy['bid_mfrr_down_x2p']).min(), 0])
    min_y2p_power = min([(strategy['bid_spot_y2p']-strategy['bid_mfrr_down_y2p']).min(), 0])
    test_cases += [(min_x2p_power, 0, RTOL_POWER, ATOL_POWER, "Only X2P power bid to the spot market can be removed")]
    test_cases += [(min_y2p_power, 0, RTOL_POWER, ATOL_POWER, "Only Y2P power bid to the spot market can be removed")]

    # --- Storage level tests ---
    # Ensure that the storage level is within the limits
    if params['name'] != 'infinite_storage':
        flow = strategy['running_p2x']*params['eff_p2x'] - strategy['running_x2p']/params['eff_x2p']
        storage_level = np.cumsum(flow) + storage_init
        storage_level_min = min([storage_level.min(), 0])
        storage_level_max = max([storage_level.max(), params['storage_size']])
        test_cases += [
            (storage_level, strategy['storage_x'], RTOL_X_LEVEL, ATOL_X_LEVEL, "The storage level must match the level implied by the bids"),
            (storage_level_min, 0, RTOL_X_LEVEL, ATOL_X_LEVEL, "The storage level must be non-negative"),
            (storage_level_max, params['storage_size'], RTOL_X_LEVEL, ATOL_X_LEVEL, "The storage level must be less than the maximal storage size")
        ]

    # Run the tests
    failed_tests = run_tests(test_cases)

    # Report the results
    if failed_tests:
        print("Some tests failed:")
        for test_desc, error_message in failed_tests:
            print(f"Test description: {test_desc}\nFailed: {error_message}")