def format_list_data(list_data: list):
    return "\n".join([f"{i+1}. {str(example)}" for i, example in enumerate(list_data)])
