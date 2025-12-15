from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, UniqueConstraint, Boolean
from sqlalchemy.orm import relationship
from database import Base

class Sector(Base):
    __tablename__ = "sectors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True) 
    ticker = Column(String, unique=True, index=True)
    type = Column(String, default="cap") # 'cap' or 'equal'

    __table_args__ = (
        UniqueConstraint('name', 'type', name='uix_sector_name_type'),
    )

    prices = relationship("PriceData", back_populates="sector", cascade="all, delete-orphan")

class PriceData(Base):
    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True, index=True)
    sector_id = Column(Integer, ForeignKey("sectors.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    close = Column(Float, nullable=False)
    
    sector = relationship("Sector", back_populates="prices")

    __table_args__ = (
        UniqueConstraint('sector_id', 'date', name='uix_sector_date'),
    )

class Constituent(Base):
    __tablename__ = "constituents"

    id = Column(Integer, primary_key=True, index=True)
    sector_id = Column(Integer, ForeignKey("sectors.id"), nullable=False)
    ticker = Column(String, unique=True, index=True)
    
    sector = relationship("Sector", back_populates="constituents")
    prices = relationship("ConstituentPrice", back_populates="constituent", cascade="all, delete-orphan")

class ConstituentPrice(Base):
    __tablename__ = "constituent_prices"

    id = Column(Integer, primary_key=True, index=True)
    constituent_id = Column(Integer, ForeignKey("constituents.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    close = Column(Float, nullable=False)
    
    # Moving Averages
    ma5 = Column(Float, nullable=True)
    ma10 = Column(Float, nullable=True)
    ma20 = Column(Float, nullable=True)
    ma50 = Column(Float, nullable=True)
    ma200 = Column(Float, nullable=True)
    
    # Flags (Is Close > MA?)
    above_ma5 = Column(Integer, nullable=True)
    above_ma10 = Column(Integer, nullable=True)
    above_ma20 = Column(Integer, nullable=True)
    above_ma50 = Column(Integer, nullable=True)
    above_ma200 = Column(Integer, nullable=True)
    
    constituent = relationship("Constituent", back_populates="prices")

    __table_args__ = (
        UniqueConstraint('constituent_id', 'date', name='uix_constituent_date'),
    )

class BreadthMetric(Base):
    __tablename__ = "breadth_metrics"

    id = Column(Integer, primary_key=True, index=True)
    sector_id = Column(Integer, ForeignKey("sectors.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    metric = Column(String, nullable=False) # e.g. 'pct_above_ma50'
    value = Column(Float, nullable=False)
    
    sector = relationship("Sector", back_populates="breadth_metrics")

    __table_args__ = (
        UniqueConstraint('sector_id', 'date', 'metric', name='uix_breadth_metric'),
    )

# Establish relationships in Sector
Sector.constituents = relationship("Constituent", back_populates="sector", cascade="all, delete-orphan")
Sector.breadth_metrics = relationship("BreadthMetric", back_populates="sector", cascade="all, delete-orphan")
