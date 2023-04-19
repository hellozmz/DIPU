import yaml
import re
from typing import Mapping, Match, Optional, Sequence
from diopi_wrapper_template import diopi_wrapper_file_template_content, diopi_wrapper_function_template_content, op_registe_template_content

class CodeTemplate:
    substitution_str = r"(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})"
    substitution = re.compile(substitution_str, re.MULTILINE)

    pattern: str
    filename: str

    @staticmethod
    def from_file(filename: str) -> "CodeTemplate":
        with open(filename, "r") as f:
            return CodeTemplate(f.read(), filename)

    def __init__(self, pattern: str, filename: str = "") -> None:
        self.pattern = pattern
        self.filename = filename

    def substitute(
        self, env: Optional[Mapping[str, object]] = None, **kwargs: object
    ) -> str:
        if env is None:
            env = {}

        def lookup(v: str) -> object:
            assert env is not None
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent: str, v: Sequence[object]) -> str:
            return "".join(
                [indent + l + "\n" for e in v for l in str(e).splitlines()]
            ).rstrip()

        def replace(match: Match[str]) -> str:
            indent = match.group(1)
            key = match.group(2)
            comma_before = ""
            comma_after = ""
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ", "
                    key = key[1:]
                if key[-1] == ",":
                    comma_after = ", "
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ", ".join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)

        return self.substitution.sub(replace, self.pattern)


def get_fun_name_from_cppsignature(cppnature):
    return re.search(r'[a-zA-Z_:]+[\w\d:]+\(' , cppnature).group().replace('(', '')


def get_op_name_from_schema(schema):
    op_name = schema[0:schema.find('(')]
    op_name = re.sub('aten::', '', op_name)
    return op_name

def create_fun_name_from_schema(schema):
    schema = schema.strip()
    op_name = schema[0:schema.find('(')]
    op_name = op_name.replace('.','_')
    op_name = "dipu_" + re.sub('aten::', '', op_name)
    op_name = op_name.lower()
    return op_name

def create_return_code_frome_schema(schema):
    return_code = schema[schema.find('->'):].replace('->', '').strip()
    return_code = re.sub('\([a-zA-Z]!\)', '&' , return_code)
    return_code = re.sub('Tensor', 'at::Tensor' , return_code)
    return_code = re.sub('\(', 'std::tuple<', return_code)
    return_code = re.sub('\)', '> ' ,return_code)
    #if return_code.find('std::tuple') >= 0:
    #    return_code = return_code.replace('Tensor&', 'Tensor')
    return return_code

def create_param_list_from_schema(schema):
    param_list = schema[schema.find('(') + 1 : schema.find('->')].strip()
    param_list = param_list[0:param_list.rfind(')')]
    param_list = re.sub('[ ]*\([a-zA-Z]!\)', '&' , param_list)
    param_list = re.sub('str\?', 'c10::optional<c10::string_view>' , param_list)
    param_list = re.sub('Tensor\?', 'const c10::optional<Tensor>&' , param_list)
    param_list = re.sub('([a-zA-Z0-9]+)\?', r'c10::optional<\1>&', param_list)
    param_list = re.sub('Tensor ', 'const Tensor& ' , param_list)
    param_list = re.sub('Scalar ', 'const Scalar& ' , param_list)
    param_list = re.sub('Tensor', 'at::Tensor' , param_list)
    param_list = re.sub('Scalar', 'at::Scalar' , param_list)
    param_list = re.sub('\*[ ,]+', '', param_list)
    param_list = re.sub('=.+,', ',', param_list)
    param_list = re.sub('=.+', '', param_list)
    param_list = re.sub(' float', ' double ', param_list)
    return param_list

def get_function_inputs_from_schema(schema):
    param_list = create_param_list_from_schema(schema)
    ins = []
    for args in param_list.split(','):
        args = args.strip()
        tensor_match_result = re.search('Tensor[ ]*&+', args)
        if tensor_match_result is not None:
            in_match_result = re.search('const[ ]+[at::]*Tensor[ &]*', args)
            if in_match_result is not None:
                ins.append(args[in_match_result.span()[1]::].strip())
        opt_tensor_match_result = re.search('const[ ]+c10::optional<at::Tensor>[ &]*([a-zA-Z_0-9]+)', args)
        if opt_tensor_match_result is not None:
            opt_tensor = re.sub('const[ ]+c10::optional<at::Tensor>[ &]*([a-zA-Z_]+)', r'\1', args).strip()
            ins.append(opt_tensor + '?')


    return ins

def get_function_outputs_from_schema(schema):
    param_list = create_param_list_from_schema(schema)
    outs = []
    for args in param_list.split(','):
        args = args.strip()
        tensor_match_result = re.search('Tensor[ ]*&+', args)
        if tensor_match_result is not None:
            in_match_result = re.search('const[ ]+[at::]*Tensor[ &]*', args)
            if in_match_result is None:
                outs.append(args[tensor_match_result.span()[1]::].strip())
    if len(outs) <= 0:
        return_param = schema[schema.find('->'):].replace('->', '').strip()
        return_param = return_param.replace('(', '')
        return_param = return_param.replace(')', '')
        params = return_param.split(',')
        if len(params) == 1 and params[0].strip() == "Tensor":
            if params[0].strip() == "Tensor":
                outs.append(f"out")
        elif len(params) > 1:
            for i in range(len(params)):
                if params[i].strip() == "Tensor":
                    outs.append(f"out{i}")

    return outs

def get_function_scalar_args_from_schema(schema):
    param_list = create_param_list_from_schema(schema)
    scalars = []
    for args in param_list.split(','):
        args = args.strip()
        scalar_match_result = re.search('Scalar[ &]*', args)
        if scalar_match_result is not None:
            scalar_param = args[scalar_match_result.span()[1]:].strip()
            scalar_param = re.sub('=.*,{1}', ',', scalar_param)
            scalar_param = re.sub('=.*', '', scalar_param)
            scalars.append(scalar_param.strip())
    return scalars


def get_function_return_param_from_schema(schema):
    return_schema= schema[schema.find('->' ) + 2:].strip()
    params = []
    return_params = return_schema.split(',')
    for i in range(len(return_params)):
        args = return_params[i]
        inplace_match = re.search('Tensor\([a-zA-Z]+!\)', args)
        pure_out_match = re.search('Tensor', args)
        if inplace_match is None and pure_out_match is not None:
            if len(return_params) > 1:
                params.append(f"out{i}")
            else:
                params.append("out")
        elif inplace_match is not None:
            arg_label = re.sub('.*(\(.*\))', r'\1',inplace_match.group())
            index = schema.find(arg_label) + len(arg_label)
            param = re.search("[a-zA-Z0-9_::]+", schema[index:]).group()
            params.append(param)

    return params

def create_call_diop_interface_code_from(schema):
    schema = schema.replace('aten::', '').strip()
    schema = schema.replace('_.', 'Inp')
    schema = schema.replace('.', '')

    outs = re.findall(",? *Tensor *\(\w+!\) *\w+", schema)[::-1]
    schema = re.sub(",? *Tensor *\(\w+!\) *\w+", '', schema)
    index = schema.find('(') + 1
    for args in outs:
        schema = schema[0:index] + args.replace(',', '') + ', ' + schema[index:]

    schema = schema.replace('(', '(ctx, ', 1)

    #re.search(",? *Tensor *\(\w+!\) *\w+")

    return_index = schema.find('->')

    if return_index > 0:
        return_args = schema[return_index + 2 :].strip()
        if re.search('Tensor[ ]*\([\w]+!\)', return_args) is None:
            return_args = re.sub('Tensor[ ]*\([\w]+!\)[ ]*', '', return_args)
            return_args = re.sub('[\(\)]', '', return_args).strip()
            outs = return_args.split(',')
            retucn_code = ''
            for i in range(len(outs)):
                retucn_code += 'out'
                if len(outs) > 1:
                    retucn_code += str(i)
                if i < len(outs) - 1:
                    retucn_code += ', '
            schema = re.sub('\([ ]*ctx', '(ctx, ' + retucn_code, schema)
    schema = schema[0 : schema.find('->')]

    for key in ['Tensor[ ]*\([\w!]+\)', 'Tensor[ ]*\?', 'Tensor[ ]*', 'bool', 'float', 'str[ ]*\?', '[,]? *\* *', '=[\w]+']:
        index = schema.find('(')
        schema = schema[0:index] +  re.sub(key , '', schema[index:])

    index = schema.find('(')
    schema = schema[0:index] +  re.sub('Scalar[ ]*' , '&', schema[index:])

    for key in ['out', '_mode', 'Tensor', '_', '[nN]{1}ative_']:
        index = schema.find('(')
        schema = re.sub(key , '', schema[:index]) + schema[index:]

    schema = 'diopi' + schema[0].upper() + schema[1:]
    schema = re.sub(' *, *', ', ', schema)
    schema = re.sub(' *, *,', ', ', schema)
    return schema


def create_cpp_signature_from_schema(schema):
    return_code = create_return_code_frome_schema(schema)
    fun_name = create_fun_name_from_schema(schema)
    param_list = create_param_list_from_schema(schema)
    cppsignature_template = CodeTemplate("$return_code $fun_name($param_list)")
    cppsignature = cppsignature_template.substitute(
        return_code=[return_code],
        fun_name=[fun_name],
        param_list=[param_list]
    )
    return cppsignature


file_template = CodeTemplate(diopi_wrapper_file_template_content)

fun_template = CodeTemplate(diopi_wrapper_function_template_content)

op_registe_template = CodeTemplate(op_registe_template_content)

def functions_code_gen(fun_config):
    if 'interface' in fun_config:
        diopi_fun_call_code = fun_config['interface'] + ";"
    else:
        diopi_interface = create_call_diop_interface_code_from(fun_config['schema'])
        diopi_fun_call_code = diopi_interface + ';'

    input_process_code = ""
    for input in get_function_inputs_from_schema(fun_config['schema']):
        if input.strip().endswith('?'):
            input = input.replace('?', '')
            input_process_code += f"\n::diopiConstTensorHandle_t {input}_diopiHandle = nullptr;\n"
            input_process_code += f"if ({input}.has_value() && {input}.value().defined()) {input}_diopiHandle = dipu::diopi_helper::toDiopiTensorHandle({input}.value());\n\n"

        else:
            input_process_code += f"::diopiConstTensorHandle_t {input}_diopiHandle = dipu::diopi_helper::toDiopiTensorHandle({input});\n"

        diopi_fun_call_code = diopi_fun_call_code.replace(input, f"{input}_diopiHandle")



    output_process_code = ""
    for output in get_function_outputs_from_schema(fun_config['schema']):
        output_process_code += f"::diopiTensorHandle_t {output}_diopiHandle = dipu::diopi_helper::toDiopiTensorHandle({output});\n"
        diopi_fun_call_code = diopi_fun_call_code.replace(output, f"{output}_diopiHandle")

    attrs_process_code = ""
    for scalar_param in get_function_scalar_args_from_schema(fun_config['schema']):
        attrs_process_code += f"::diopiScalar_t {scalar_param}_diopiScalar = dipu::diopi_helper::toDiopiScalar({scalar_param});\n";
        diopi_fun_call_code = diopi_fun_call_code.replace(scalar_param, f"{scalar_param}_diopiScalar")

    return_code = ""
    return_param = get_function_return_param_from_schema(fun_config['schema'])
    if len(return_param) == 0:
        return_code = "return;\n"
    elif len(return_param) == 1:
        return_code = f"return {return_param[0]};\n"
    else:
        params = ''
        for i in range(len(return_param)):
            params += return_param[i]
            if i < len(return_param) - 1:
                params += ', '
        return_code = f"return std::tie({params});"

    fbody = fun_template.substitute(
            comment=[fun_config['schema']],
            cppsignautre=[create_cpp_signature_from_schema(fun_config['schema'])],
            custom_code=[fun_config.get('custom_code', '').replace('; ', ';\n')],
            input_process_code=[input_process_code],
            output_process_code=[output_process_code],
            diopi_fun_call_code=[diopi_fun_call_code],
            attrs_process_code=[attrs_process_code],
            return_code=[return_code],
    )
    diopi_interface = fun_config.get('interface', create_call_diop_interface_code_from(fun_config['schema']))
    registe_body = op_registe_template.substitute(
            register_name=[get_op_name_from_schema(fun_config['schema'])],
            aten_fun_name=['dipu::native::' + create_fun_name_from_schema(fun_config['schema'])],
            diopi_fun_name=[get_fun_name_from_cppsignature(diopi_interface).replace('diopi', '::diopi')],
    )
    return fbody, registe_body


def parase_args():
    import argparse
    parser = argparse.ArgumentParser(description='autogen diopi wrapper code')
    parser.add_argument('--config', type=str, default = 'diopi_functions.yaml', help='path to functions config file')
    parser.add_argument('--out', type=str, default = 'AutoGenedKernels.cpp', help='path to functions config file')

    args = parser.parse_args()
    return args

def main():
    args = parase_args()

    with open(args.config) as diopi_functions_file:
        file_data = diopi_functions_file.read()
        funcs_config = yaml.load(file_data, Loader=yaml.FullLoader)


    functions_code = ''
    op_registe_code = ''

    for fun_config in funcs_config:
        fun_code, register_code = functions_code_gen(fun_config)
        functions_code += fun_code
        op_registe_code += register_code

    autogened_file = file_template.substitute(
        functions_code=[functions_code],
        op_registe_code=[op_registe_code]
    )
    autogened_file = re.sub('\n\n\n+', '\n', autogened_file)
    autogened_file = re.sub('[ ]*,[ ]*', ', ', autogened_file)
    with open(args.out, 'w') as cpp_file:
        cpp_file.write(autogened_file)

    print(f"Successfully generate {args.out} according to the configuration file {args.config}")


if __name__ == "__main__":
    main()