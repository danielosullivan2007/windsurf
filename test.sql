

with base as (
    select playerid, casinogametypeid, sum(theoreticalcentsusd) as total_theoretical
    from boris.dbo.AGG_CasinoBet
    where DateKey > '20250401'
    group by playerid, casinogametypeid
)
select CasinoGame, Gamevariant,  sum(total_theoretical) as total_theoretical, count(distinct playerid) as player_count
from base b
join boris.dbo.dim_CasinoGameType g
on b.casinogametypeid = g.casinogametypeid
group by CasinoGame, Gamevariant




with
    base
    as
    (
        select playerid, casinogametypeid, sum(theoreticalcentsusd) as total_theoretical
        from boris.dbo.AGG_CasinoBet with (nolock)
        where DateKey > '20250507'
        group by playerid, casinogametypeid
    )
select g.CasinoGame, g.Gamevariant, sum(b.total_theoretical) as total_theoretical, count(distinct b.playerid) as player_count
from base b
    join boris.dbo.dim_CasinoGameType g with (nolock)
    on b.casinogametypeid = g.casinogametypeid
group by g.CasinoGame, g.Gamevariant

select top 1 * from boris.dbo.AGG_CasinoBet
where DateKey > '20250507'