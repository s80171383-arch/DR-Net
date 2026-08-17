from pytorch.train_isprs import parse_args, profile_batches


REQUIRED_ARGS = ["--config", "config.yaml", "--data-root", "data", "--output-dir", "out"]


def test_profile_steps_argument_parses():
    assert parse_args(REQUIRED_ARGS + ["--profile-steps", "3"]).profile_steps == 3


def test_profile_mode_yields_exact_requested_number_of_steps():
    assert [batch for batch, _ in profile_batches(range(10), 3)] == [0, 1, 2]


def test_zero_profile_steps_preserves_full_training_iteration():
    assert [batch for batch, _ in profile_batches(range(5), 0)] == []
    assert parse_args(REQUIRED_ARGS).profile_steps == 0
