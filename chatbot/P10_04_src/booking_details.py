# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List



class BookingDetails:
    def __init__(
        self,
        destination: str = None,
        origin: str = None,
        travel_start_date: str = None,
        travel_end_date: str = None,
        budget: str = None,
        unsupported_airports: List[str] = None,
    ):
        self.destination = destination
        self.origin = origin
        self.travel_start_date = travel_start_date
        self.travel_end_date = travel_end_date
        self.budget = budget
        self.unsupported_airports = unsupported_airports or []
