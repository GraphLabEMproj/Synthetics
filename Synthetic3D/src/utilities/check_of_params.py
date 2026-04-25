def check_param(params, name_of_param, set_default_value=None, preffix="\t"):
    if not name_of_param in params.keys():
        params[name_of_param] = set_default_value
        return [f'{preffix}WARNING!!! "{name_of_param}" param don\'t set! Will be set as {set_default_value}']
    else:
        return []
