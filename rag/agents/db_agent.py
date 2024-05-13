def prices_retrieval_tool():
    """
    This function will help you find the price of a single inventory.
    :return: the price
    """
    def get_price(inventory_item: str) -> str:
        # json_agent = get_json_agent("./inventory_prices_dict.json")
        # result = json_agent_executor.run(
        #     f"""get the price of {inventory_item} from the json file.
        #     Find the closest match to the item you're looking for in that json, e.g.
        #      if you're looking for "mahogany oak table" and that is not in the json, use "table".
        #     Be mindful of the format of the json - there is no list that you can access via [0], so don't try to do that
        #     """)
        
        # return a quote that includes the string str 
        result = f"The price of the inventory item {inventory_item} is $42.42."
        return result

    price_tool = StructuredTool.from_function(func=get_price,
                                              name='get inventory and furniture prices',
                                              description='This function will help you get the price of an inventory or'
                                                          ' furniture item.')

    return price_tool