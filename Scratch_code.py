Network
    generate: construct nodes
        locations = [[ [1,2], [2,3] ], [[1,0],  [-1,1]]] 3-dimensional list [ 2 by n matrix, matrix, ...]
        rates = 2-dimensional list 2 - dimensional list [vector, vector, ...]
        data_type_dist: 1-dimnesional np.array
        layer_dic: dictionary
        A: 3-dimensional list [matrix, matrix, matrix]
        medium

    update_time: return close_event_time
        close_event_time = self.network_nodes[0][0].remaining_time
        for l in range(layer_num):
            for i in range(node_num):
                if close_event_time > self.network_nodes[l][i].remaining_time:
                    close_event_time = self.network_nodes[l][i].remaining_time
        if medium.remaining_time != [] and medium.remaining_time < close_event_time:
            close_event_time = medium.remaining_time
        return close_event_time

    update:
        for l in range(layer_num):
            for i in range(node_num):
                sending_node = network_node[l][i]
                if sending_node.data_stack == []:
                    pass
                elif sending_node.data_stack != [] and close_event_time == sending_node.remaining_time:
                    sending_data = sending_node.data_stack[0]
                    check end node
                    if sending_node == end_node:
                        increase num_completed_data
                        remove completed_data
                    else:
                        find next_node
                        transfer (sending_data, next_node)

                    source: send data to medium, generate new data
                        sending_data = source.data_stack[0]
                        transfer (sending_data, end_node) to medium
                        remove the sent data
                        generate new_data, remaining_time
                        add new_data
                    server: send data to medium
                        sending_data = server.data_stack[0]
                        if server == last server of the data:
                            increase num_completed by one
                        else:
                            transfer (sending_data, end_node) to medium
                        remove the sent data (or the completed data)
                        update remaining_time
                            if server.data_stack == []:
                                reamining_time = []
                            else:
                                remaining_time = choice(server.rate)
        medium:
            if self.remaining_time == close_event_time:
                Handover to next_node
                transfer data