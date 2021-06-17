class Scatter(object):

    def __init__(self, host_variable, guest_variable):
        """
        scatter values from guest and hosts

        Args:
            host_variable: a variable represents `Host -> Arbiter`
            guest_variable: a variable represent `Guest -> Arbiter`

        Examples:

            >>> from federatedml.framework.homo.util import scatter
            >>> s = scatter.Scatter(host_variable, guest_variable)
            >>> for v in s.get():
                    print(v)


        """
        self._host_variable = host_variable
        self._guest_variable = guest_variable

    def get(self, suffix=tuple(), host_ids=None):
        """
        create a generator of values from guest and hosts.

        Args:
            suffix: tag suffix
            host_ids: ids of hosts to get value from.
                If None provided, get values from all hosts.
                If a list of int provided, get values from all hosts listed.

        Returns:
            a generator of scatted values

        Raises:
            if host_ids is neither None nor a list of int, ValueError raised
        """
        yield self._guest_variable.get(idx=0, suffix=suffix) \
              + ("guest",) + (0,)
        if host_ids is None:
            host_ids = -1

        if host_ids == -1:
            src_parties = self._host_variable.roles_to_parties(roles=["host"])
            host_ids = [i for i in range(len(src_parties))]

        for host_id in host_ids:
            yield self._host_variable.get(idx=host_id, suffix=suffix) \
                  + ("host",) + (host_id,)
