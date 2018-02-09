/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Close the bidding for a loan listing and choose the
 * lowest bid that is below the asking rate
 * @param {org.marnet.loan.auction.CloseBidding} closeBidding - the closeBidding transaction
 * @transaction
 */
function closeBidding(closeBidding) {
    var listing = closeBidding.listing;
    if(listing.state !== 'FOR_SALE'){
        throw new Error('Listing is not FOR SALE');
    }
    // by default we mark the listing as RESERVE_NOT_MET
    listing.state = 'RESERVE_NOT_MET';
    var lowestOffer = null;
    var buyer = null;
    var seller = null;
    if(listing.offers && listing.offers.length > 0){
        //sort the bids in desc order
        listing.offers.sort(function(a,b){
            return (a.bidRoi - b.bidRoi);
        });
        lowestOffer = listing.offers[0];
        if(lowestOffer.bidRoi <= listing.reserveRoi) {
            listing.state = 'SOLD';
            buyer = lowestOffer.investor;
            seller = listing.loan.cdoOwner;

            listing.loan.cdoOwner = buyer;
            listing.offers = null;
        }
    }
    return getAssetRegistry('org.marnet.loan.auction.Loan')
        .then(function(loanRegistry){
            //save the loan
            if(lowestOffer){
                return loanRegistry.update(listing.loan);
            } else {
                return true;
            }
        })
        .then(function(){
            return getAssetRegistry('org.marnet.loan.auction.LoanListing')
        })
        .then(function(loanListingRegistry){
            //save the loan listing
            return loanListingRegistry.update(listing);
        })
  		.then(function() {
            return getParticipantRegistry('org.marnet.loan.auction.Investor')
        })
        .then(function(userRegistry){
            //save the buyer
            if(listing.state == 'SOLD') {
                return userRegistry.updateAll([buyer]);
            } else {
                return true;
            }
        });

}

/**
 * Make an Offer for a LoanListing
 * @param {org.marnet.loan.auction.Offer} offer - the offer
 * @transaction
 */

function makeOffer(offer) {
    var listing = offer.listing;
    if(listing.state !== 'FOR_SALE') {
        throw new Error('Listing is not FOR SALE');
    }
    if(listing.offers == null) {
        listing.offers = [];
    }
    listing.offers.push(offer);
    return getAssetRegistry('org.marnet.loan.auction.LoanListing')
        .then(function(loanListingRegistry){
            //save the loan listing
            return loanListingRegistry.update(listing);
        });
}