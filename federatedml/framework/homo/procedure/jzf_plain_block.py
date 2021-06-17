from arch.api.utils import log_utils
from federatedml.util import consts
from federatedml.secureprotol.jzf_quantize import QuantizingClient, QuantizingArbiter

LOGGER = log_utils.getLogger()


class Arbiter(object):

    def __init__(self, secure_aggregate_args):
        super(Arbiter, self).__init__()
        self.arbiter_to_guest = None
        self.arbiter_to_host = None
        self.guest_to_arbiter = None
        self.host_to_arbiter = None

        self.do_quantize = False
        if 'quantize' in secure_aggregate_args:
            self.do_quantize = True
            self.int_bits = secure_aggregate_args['quantize']['int_bits']
            self.batch = secure_aggregate_args['quantize']['batch']
            self.element_bits = secure_aggregate_args['quantize']['element_bits']
            self.secure = secure_aggregate_args['quantize']['secure']

        self.quantizer = None
        self.has_set_layer_size_list = False

    def set_iter_index(self, iter_index):
        if self.do_quantize:
            self.quantizer.set_iter(iter_index)

    def register_plain_cipher(self, transfer_variables):
        self.arbiter_to_guest = transfer_variables.arbiter_to_guest
        self.arbiter_to_host = transfer_variables.arbiter_to_host
        self.guest_to_arbiter = transfer_variables.guest_to_arbiter
        self.host_to_arbiter = transfer_variables.host_to_arbiter

        return self

    def create_quantizer(self):
        if self.do_quantize:
            client_cnt = 1  # 1 for guest
            host_list = self.arbiter_to_host.roles_to_parties([consts.HOST])
            client_cnt += len(host_list)

            self.quantizer = QuantizingArbiter(self.int_bits,
                                               self.arbiter_to_guest,
                                               self.guest_to_arbiter,
                                               self.arbiter_to_host,
                                               self.host_to_arbiter,
                                               self.batch,
                                               self.element_bits,
                                               self.secure)
            self.quantizer.broadcast_num_clients(client_cnt)

        return self

    def help_quantize(self):
        if self.do_quantize:
            if not self.has_set_layer_size_list:
                self.quantizer.set_layer_size_list()
                self.has_set_layer_size_list = True
            return self.quantizer.help_quantize()


class _Client(object):

    def __init__(self, args):
        super(_Client, self).__init__()
        self.cipher = None
        self.quantizer = None
        self.do_quantize = False

        if 'quantize' in args:
            self.do_quantize = True
            self.int_bits = args['quantize']['int_bits']
            self.batch = args['quantize']['batch']
            self.element_bits = args['quantize']['element_bits']
            self.padding = args['quantize']['padding']
            self.secure = args['quantize']['secure']

    def set_iter_index(self, iter_index):
        if self.do_quantize:
            self.quantizer.set_iter(iter_index)

    def unquantize(self, weights):
        if self.do_quantize:
            return self.quantizer.unquantize(weights)
        else:
            return weights

    def unnormalize(self, weights):
        if self.do_quantize:
            return self.quantizer.unnormalize(weights)
        else:
            return weights


class Guest(_Client):

    def __init__(self, secure_aggregate_args):
        super(Guest, self).__init__(secure_aggregate_args)
        self.guest_to_arbiter = None
        self.arbiter_to_guest = None
        self.has_send_layer_size_list = False

    def create_quantizer(self):
        if self.do_quantize:
            self.quantizer = QuantizingClient(self.int_bits,
                                              self.arbiter_to_guest,
                                              self.guest_to_arbiter,
                                              self.batch,
                                              self.element_bits,
                                              self.padding,
                                              self.secure)
            _ = self.quantizer.receive_num_clients()

        return self

    def register_plain_cipher(self, transfer_variables):
        self.guest_to_arbiter = transfer_variables.guest_to_arbiter
        self.arbiter_to_guest = transfer_variables.arbiter_to_guest
        return self

    def quantize(self, weights):
        if self.do_quantize:
            if not self.has_send_layer_size_list:
                self.quantizer.send_layer_size_list(weights)
                self.has_send_layer_size_list = True
            return self.quantizer.quantize(weights)
        else:
            return weights

    def normalize(self, weights):
        if self.do_quantize:
            if not self.has_send_layer_size_list:
                self.quantizer.send_layer_size_list(weights)
                self.has_send_layer_size_list = True
            return self.quantizer.normalize(weights)
        else:
            return weights


class Host(_Client):

    def __init__(self, secure_aggregate_args):
        super(Host, self).__init__(secure_aggregate_args)
        self.arbiter_to_host = None
        self.host_to_arbiter = None
        self.guest_uuid = None
        self.has_set_layer_size_list = False

    def create_quantizer(self):
        if self.do_quantize:
            self.quantizer = QuantizingClient(self.int_bits,
                                              self.arbiter_to_host,
                                              self.host_to_arbiter,
                                              self.batch,
                                              self.element_bits,
                                              self.padding,
                                              self.secure)
            _ = self.quantizer.receive_num_clients()

        return self

    def register_plain_cipher(self, transfer_variables):
        self.host_to_arbiter = transfer_variables.host_to_arbiter
        self.arbiter_to_host = transfer_variables.arbiter_to_host
        return self

    def quantize(self, weights):
        if self.do_quantize:
            if not self.has_set_layer_size_list:
                self.quantizer.set_layer_size_list(weights)
                self.has_set_layer_size_list = True
            return self.quantizer.quantize(weights)
        else:
            return weights

    def normalize(self, weights):
        if self.do_quantize:
            if not self.has_set_layer_size_list:
                self.quantizer.set_layer_size_list(weights)
                self.has_set_layer_size_list = True
            return self.quantizer.normalize(weights)
        else:
            return weights
