class AdaptiveRouter:
    def route_order(self, order: Dict) -> OrderRoutingPlan:
        """Enhanced with circuit breaker pattern"""
        if self._circuit_breaker_tripped():
            raise CircuitBreakerTripped()
            
        with self._lock:
            self._validate_order(order)
            venues = self._select_venues(order)
            return self._create_plan(venues, order)
    
    def _select_venues(self, order: Dict) -> List[ExecutionVenue]:
        """Multi-criteria decision making with fallbacks"""
        venues = self._filter_venues(order)
        if not venues:
            venues = self._fallback_venues(order)
        return sorted(venues, key=self._venue_score, reverse=True)
    
    def _venue_score(self, venue: ExecutionVenue) -> float:
        """Dynamic scoring with market conditions"""
        base_score = self.venue_scores[venue.name]
        market_impact = self._calculate_market_impact(venue)
        return base_score * (1 - market_impact)
