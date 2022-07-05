from environ.nsl_kdd import nsl_kdd_env


def test_can_be_created():
    env = nsl_kdd_env(1, 1)
    assert env is not None
    assert env.reward == 1
    assert env.child_id == 1


def test_creates_correct_file_path_for_child_one():
    env = nsl_kdd_env(1, 1)
    assert env.get_file_path() == "../..DataSets/NSL-KDD/Small_Splits//KDDTrain+_1.txt"  # noqa: E501


def test_creates_correct_file_path_for_child_two():
    env = nsl_kdd_env(1, 2)
    assert env.get_file_path() == "../..DataSets/NSL-KDD/Small_Splits//KDDTrain+_2.txt"  # noqa: E501


# def test_preprocesses_sample_dataset_correctly(mocker):
#     env = nsl_kdd_env(1, 1)
#     # Mock file_system function and return a test dataframe
#     mocker.patch('environ.file_system.sample_csv')
#     mocker.result_value = pd.DataFrame()
#     env.generate_dataframes()

#     assert env.lookup_df is not None
